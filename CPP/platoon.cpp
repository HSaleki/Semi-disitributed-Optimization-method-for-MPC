// Closed-loop platoon MPC on the distributed active-set QP solver.
//
// z_i = [u0; x1; u1; x2; ... ; u_{N-1}; xN],  x = [p; v; a],  u = jerk.
// Followers track the leader at a fixed reference gap; consecutive vehicles are
// coupled by  p_i(k) - p_{i-1}(k) <= -D_hard.

#include "dmpc.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace dmpc;

namespace {

struct Bounds {
    double uMin = -2, uMax = 2, vMin = 0, vMax = 30, aMin = -3, aMax = 3;
};

inline int iu(int k, int nu, int nx) { return k * (nu + nx); }
inline int ix(int k, int nu, int nx) { return k * (nu + nx) + nu; }

void chain3rd(double Ts, Mat& A, Mat& B) {
    A.resize(3, 3);
    A << 1, Ts, 0.5 * Ts * Ts, 0, 1, Ts, 0, 0, 1;
    B.resize(3, 1);
    B << Ts * Ts * Ts / 6.0, Ts * Ts / 2.0, Ts;
}

// discrete-time LQR by iterating the Riccati recursion to a fixed point
void dlqr(const Mat& A, const Mat& B, const Mat& Q, const Mat& R, Mat& K, Mat& P) {
    P = Q;
    for (int it = 0; it < 10000; ++it) {
        const Mat S = R + B.transpose() * P * B;
        const Mat Knew = S.ldlt().solve(B.transpose() * P * A);
        const Mat Pnew =
            Q + A.transpose() * P * A - A.transpose() * P * B * Knew;
        const double diff = (Pnew - P).cwiseAbs().maxCoeff();
        P = Pnew;
        K = Knew;
        if (diff < 1e-14 * std::max(1.0, P.cwiseAbs().maxCoeff())) break;
    }
}

void buildDynamics(const Mat& Ad, const Mat& Bd, int N, Mat& Aeq, Mat& Beq) {
    const int nx = 3, nu = 1;
    Aeq = Mat::Zero(N * nx, N * (nu + nx));
    Beq = Mat::Zero(N * nx, nx);
    for (int k = 0; k < N; ++k) {
        Aeq.block(k * nx, ix(k, nu, nx), nx, nx) = Mat::Identity(nx, nx);
        Aeq.block(k * nx, iu(k, nu, nx), nx, nu) = -Bd;
        if (k >= 1)
            Aeq.block(k * nx, ix(k - 1, nu, nx), nx, nx) = -Ad;
        else
            Beq.block(0, 0, nx, nx) = Ad;
    }
}

Mat buildHessian(const Mat& Q, const Mat& R, const Mat& P, int N) {
    const int nx = 3, nu = 1;
    Mat H = Mat::Zero(N * (nu + nx), N * (nu + nx));
    for (int k = 0; k < N; ++k) {
        H.block(iu(k, nu, nx), iu(k, nu, nx), nu, nu) = 2 * R;
        H.block(ix(k, nu, nx), ix(k, nu, nx), nx, nx) = 2 * (k == N - 1 ? P : Q);
    }
    return H;
}

void buildLocalIneq(const Bounds& bd, int N, Mat& Aloc, Vec& bloc) {
    const int nx = 3, nu = 1, nZ = N * (nu + nx);
    Mat Eu = Mat::Zero(N, nZ), Ev = Mat::Zero(N, nZ), Ea = Mat::Zero(N, nZ);
    for (int k = 0; k < N; ++k) {
        Eu(k, iu(k, nu, nx)) = 1.0;
        Ev(k, ix(k, nu, nx) + 1) = 1.0;
        Ea(k, ix(k, nu, nx) + 2) = 1.0;
    }
    Aloc.resize(6 * N, nZ);
    Aloc << Eu, -Eu, Ev, -Ev, Ea, -Ea;
    bloc.resize(6 * N);
    bloc << bd.uMax * Vec::Ones(N), -bd.uMin * Vec::Ones(N),
        bd.vMax * Vec::Ones(N), -bd.vMin * Vec::Ones(N), bd.aMax * Vec::Ones(N),
        -bd.aMin * Vec::Ones(N);
}

void buildCoupling(int M, int N, double D, const Vec& leaderP,
                   std::vector<Mat>& Acell, Vec& b) {
    const int nx = 3, nu = 1, nZ = N * (nu + nx);
    Mat Ep = Mat::Zero(N, nZ);
    for (int k = 0; k < N; ++k) Ep(k, ix(k, nu, nx)) = 1.0;

    Acell.assign(M, Mat::Zero(M * N, nZ));
    b = -D * Vec::Ones(M * N);

    Acell[0].topRows(N) = Ep;
    b.head(N) = leaderP.tail(N).array() - D;
    for (int e = 1; e < M; ++e) {
        Acell[e].middleRows(e * N, N) += Ep;
        Acell[e - 1].middleRows(e * N, N) -= Ep;
    }
}

Vec buildZref(const Mat& xref, const Mat& uref) {
    const int nx = static_cast<int>(xref.rows());
    const int N = static_cast<int>(xref.cols());
    const int nu = static_cast<int>(uref.rows());
    Vec z = Vec::Zero(N * (nu + nx));
    for (int k = 0; k < N; ++k) {
        z.segment(iu(k, nu, nx), nu) = uref.col(k);
        z.segment(ix(k, nu, nx), nx) = xref.col(k);
    }
    return z;
}

// Time-shift the previous solution and pad the tail with the clipped LQR law.
// The terminal stage has no preimage under the shift, so the pad solves the
// scalar QP over all six stage-N rows; the multiplier of the binding row is
// handed back so the dual shift stays consistent.
void shiftZ(const Mat& Ad, const Mat& Bd, const std::vector<Vec>& Zprev,
            const Vec& leaderP, double Dref, const Mat& K, double uRef,
            const Bounds& bd, double R, int N, const Mat& P,
            std::vector<Vec>& out, std::vector<int>& actN, std::vector<double>& muN,
            Mat& lamCorr) {
    const int nx = 3, nu = 1;
    const Eigen::RowVector3d Cv(0, 1, 0), Ca(0, 0, 1);
    const double Reff = R + (Bd.transpose() * P * Bd)(0, 0);

    Vec cc(6);
    cc << 1.0, -1.0, (Cv * Bd)(0, 0), -(Cv * Bd)(0, 0), (Ca * Bd)(0, 0),
        -(Ca * Bd)(0, 0);
    Mat gx = Mat::Zero(nx, 6);
    gx.col(2) = Cv.transpose();
    gx.col(3) = -Cv.transpose();
    gx.col(4) = Ca.transpose();
    gx.col(5) = -Ca.transpose();

    const int M = static_cast<int>(Zprev.size());
    out.assign(M, Vec());
    actN.assign(M, -1);
    muN.assign(M, 0.0);
    lamCorr = Mat::Zero(nx, M);

    for (int i = 0; i < M; ++i) {
        const Vec& Z = Zprev[i];
        Vec Zs = Vec::Zero(Z.size());
        Zs.head(Z.size() - (nx + nu)) = Z.tail(Z.size() - (nx + nu));

        const Vec xNm1 = Z.tail(nx);
        Vec xr(nx);
        xr << leaderP(leaderP.size() - 2) - (i + 1) * Dref, 20.0, 0.0;
        const double uLqr = uRef - (K * (xNm1 - xr))(0);

        const Vec xfree = Ad * xNm1;
        Vec dd(6);
        dd << bd.uMax, -bd.uMin, bd.vMax - (Cv * xfree)(0),
            -bd.vMin + (Cv * xfree)(0), bd.aMax - (Ca * xfree)(0),
            -bd.aMin + (Ca * xfree)(0);

        int hiJ = -1, loJ = -1;
        double hi = INF, lo = -INF;
        for (int j = 0; j < 6; ++j) {
            const double v = dd(j) / cc(j);
            if (cc(j) > 0 && v < hi) {
                hi = v;
                hiJ = j;
            }
            if (cc(j) < 0 && v > lo) {
                lo = v;
                loJ = j;
            }
        }

        int kr = -1;
        double uPad;
        if (lo > hi) {
            uPad = std::min(std::max(uLqr, bd.uMin), bd.uMax);
        } else if (uLqr > hi) {
            uPad = hi;
            kr = hiJ;
        } else if (uLqr < lo) {
            uPad = lo;
            kr = loJ;
        } else {
            uPad = uLqr;
        }

        if (kr >= 0) {
            actN[i] = (kr + 1) * N - 1;          // stage-N row of block kr
            muN[i] = 2 * Reff * (uLqr - uPad) / cc(kr);
            lamCorr.col(i) = -gx.col(kr) * muN[i];
        }

        Zs.tail(nx + nu)(0) = uPad;
        Zs.tail(nx) = Ad * xNm1 + Bd * uPad;
        out[i] = Zs;
    }
}

// blocks are contiguous, N entries each; the terminal entry is held
template <typename T>
std::vector<T> shiftBlocks(const std::vector<T>& v, int N) {
    std::vector<T> out(v.size());
    const int nb = static_cast<int>(v.size()) / N;
    for (int blk = 0; blk < nb; ++blk) {
        for (int k = 0; k < N - 1; ++k) out[blk * N + k] = v[blk * N + k + 1];
        out[blk * N + N - 1] = v[blk * N + N - 1];
    }
    return out;
}

Vec shiftBlocksVec(const Vec& v, int N) {
    Vec out = Vec::Zero(v.size());
    const int nb = static_cast<int>(v.size()) / N;
    for (int blk = 0; blk < nb; ++blk) {
        out.segment(blk * N, N - 1) = v.segment(blk * N + 1, N - 1);
        out(blk * N + N - 1) = v(blk * N + N - 1);
    }
    return out;
}

std::vector<Vec> shiftDualLocal(const std::vector<Vec>& dual, int N, int nx,
                                int neq, const Mat& P,
                                const std::vector<Vec>& z0, const Vec& leaderP,
                                double Dref, double vRef, const Mat& lamCorr) {
    std::vector<Vec> out;
    for (size_t i = 0; i < dual.size(); ++i) {
        Vec xr(nx);
        xr << leaderP(leaderP.size() - 1) - static_cast<double>(i + 1) * Dref,
            vRef, 0.0;
        const Vec xN = z0[i].tail(nx);

        Vec lamNew = Vec::Zero(neq);
        lamNew.head(neq - nx) = dual[i].segment(nx, neq - nx);
        lamNew.tail(nx) = -2 * P * (xN - xr) + lamCorr.col(i);

        const Vec muNew = shiftBlocksVec(dual[i].tail(dual[i].size() - neq), N);
        Vec d(neq + muNew.size());
        d << lamNew, muNew;
        out.push_back(d);
    }
    return out;
}

struct Args {
    int steps = 10, N = 20, M = 5;
    double Ts = 1.0;
    bool quiet = false;
    std::string csv;
};

Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : nullptr; };
        if (s == "-h" || s == "--help") {
            std::printf(
                "usage: platoon [steps] [-N horizon] [-M agents] [--ts dt]\n"
                "               [--csv PATH] [--quiet]\n"
                "\n"
                "  steps      closed-loop steps to simulate (default 50)\n"
                "  -N         prediction horizon (default 20)\n"
                "  -M         number of followers (default 5)\n"
                "  --ts       sample time in seconds (default 1.0)\n"
                "  --csv      write the trajectory to PATH for plotting\n"
                "  --quiet    summary only\n");
            std::exit(0);
        } else if (s == "-N") {
            a.N = std::atoi(next());
        } else if (s == "-M") {
            a.M = std::atoi(next());
        } else if (s == "--ts") {
            a.Ts = std::atof(next());
        } else if (s == "--csv") {
            a.csv = next();
        } else if (s == "--quiet") {
            a.quiet = true;
        } else {
            a.steps = std::atoi(argv[i]);
        }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const Args args = parseArgs(argc, argv);
    const int N = args.N, M = args.M, nx = 3, nu = 1;
    const double Ts = args.Ts;
    const double Dhard = 30.0, Dref = 35.0;   // Dref > Dhard, or every coupled row
    const double vRef = 20.0, uRef = 0.0;     // is weakly active and the ratio
                                              // tests degenerate
    const Bounds bd;

    Mat Qs = Mat::Zero(3, 3);
    Qs.diagonal() << 0.3, 0.6, 0.9;
    Mat Rs = Mat::Ones(1, 1);

    Mat Ad, Bd, Aeq, Beq, K, P;
    chain3rd(Ts, Ad, Bd);
    buildDynamics(Ad, Bd, N, Aeq, Beq);
    dlqr(Ad, Bd, Qs, Rs, K, P);
    const Mat Hess = buildHessian(Qs, Rs, P, N);

    Vec leaderP = Vec::LinSpaced(N + 1, 0.0, N * Ts * vRef);
    Mat Aloc;
    Vec bloc;
    buildLocalIneq(bd, N, Aloc, bloc);
    const int neq = static_cast<int>(Aeq.rows());
    const int nineq = static_cast<int>(Aloc.rows());

    const double seedIC[5][3] = {{-60, 18, -1}, {-110, 25, 1}, {-170, 15, 0},
                                 {-240, 12, -1}, {-280, 5, 1}};
    std::vector<Vec> x(M, Vec::Zero(nx));
    for (int i = 0; i < M; ++i) {
        if (i < 5)
            x[i] << seedIC[i][0], seedIC[i][1], seedIC[i][2];
        else
            x[i] << -280.0 - 55 * (i - 4), 10.0, 0.0;
    }

    std::vector<Vec> z0(M, Vec::Zero(N * (nx + nu)));
    std::vector<Mask> locActive(M, Mask(nineq, 0));
    std::vector<Vec> locDual(M, Vec::Zero(neq + nineq));
    Mask coupActive(M * N, 0);
    Vec coupDual = Vec::Zero(M * N);

    Logger log(M);
    std::vector<Vec> Zprev;
    std::ofstream csv;
    if (!args.csv.empty()) {
        csv.open(args.csv);
        csv << "t,agent,pos,vel,acc,rho,iter,nAc,central,local\n";
    }

    double worstRho = 0.0;
    long long totalIters = 0, maxIter = 0;

    for (int t = 0; t < args.steps; ++t) {
        std::vector<Mat> Acoup;
        Vec bcoup;
        buildCoupling(M, N, Dhard, leaderP, Acoup, bcoup);
        log.reset();

        Problem prob;
        prob.nodes.resize(M);
        for (int i = 0; i < M; ++i) {
            Mat xref(3, N);
            for (int k = 0; k < N; ++k)
                xref.col(k) << leaderP(k + 1) - (i + 1) * Dref, vRef, 0.0;

            NodeData& nd = prob.nodes[i];
            nd.Q = Hess;
            nd.p = -Hess * buildZref(xref, Mat::Zero(nu, N));
            nd.G = Aeq;
            nd.g = Beq * x[i];
            nd.H = Aloc;
            nd.h = bloc;
            nd.A = Acoup[i];
        }

        if (!Zprev.empty()) {
            std::vector<int> actN;
            std::vector<double> muN;
            Mat lamCorr;
            shiftZ(Ad, Bd, Zprev, leaderP, Dref, K, uRef, bd, Rs(0, 0), N, P, z0,
                   actN, muN, lamCorr);

            for (int i = 0; i < M; ++i) locActive[i] = shiftBlocks(locActive[i], N);
            locDual = shiftDualLocal(locDual, N, nx, neq, P, z0, leaderP, Dref,
                                     vRef, lamCorr);

            for (int i = 0; i < M; ++i) {
                for (int blk = 1; blk <= 6; ++blk) {
                    const int row = blk * N - 1;   // stage-N row of each block
                    locActive[i][row] = 0;         // shiftZ owns the stage-N block
                    locDual[i](neq + row) = 0.0;
                }
                if (actN[i] >= 0) {
                    locActive[i][actN[i]] = 1;
                    locDual[i](neq + actN[i]) = muN[i];
                }
            }

            coupActive = shiftBlocks(coupActive, N);
            coupDual = shiftBlocksVec(coupDual, N);

            Vec Ax = Vec::Zero(M * N);
            for (int i = 0; i < M; ++i) Ax += Acoup[i] * z0[i];
            const double tol = 10 * EPS * std::max(ninf(Ax), ninf(bcoup));
            for (int e = 1; e <= M; ++e) {
                const int row = e * N - 1;
                coupActive[row] = (Ax(row) - bcoup(row)) > tol * Dhard ? 1 : 0;
                coupDual(row) = 0.0;
            }
        }

        for (int i = 0; i < M; ++i) {
            prob.nodes[i].z0 = z0[i];
            prob.nodes[i].locActive = locActive[i];
            prob.nodes[i].locDual = locDual[i];
        }
        prob.b = bcoup;
        prob.coupActive = coupActive;
        prob.coupDual = coupDual;

        const Solution sol = solve(prob, log);
        if (!sol.converged) {
            std::fprintf(stderr, "step %d: solver did not converge\n", t);
            return 1;
        }

        const double rho = kktResidual(prob, sol);
        worstRho = std::max(worstRho, rho);
        totalIters += sol.iter;
        maxIter = std::max<long long>(maxIter, sol.iter);

        if (!args.quiet)
            std::printf("t=%3d  K=%4d  rho=%.3e  |A_c|=%3d  central=%7lld  local=%8lld\n",
                        t, sol.iter, rho, count(sol.activeC), log.centralTotal(),
                        log.localTotal());

        if (csv.is_open()) {
            for (int i = 0; i < M; ++i)
                csv << t << ',' << i << ',' << x[i](0) << ',' << x[i](1) << ','
                    << x[i](2) << ',' << rho << ',' << sol.iter << ','
                    << count(sol.activeC) << ',' << log.centralTotal() << ','
                    << log.localTotal() << '\n';
        }

        Zprev = sol.z;
        locActive = sol.activeL;
        locDual = sol.dualL;
        coupActive = sol.activeC;
        coupDual = sol.dualC;

        for (int i = 0; i < M; ++i) x[i] = Ad * x[i] + Bd * Zprev[i](0);

        Vec shifted(N + 1);
        shifted.head(N) = leaderP.tail(N);
        shifted(N) = leaderP(N) + Ts * vRef;
        leaderP = shifted;
    }

    std::printf("\nworst rho over %d steps: %.3e\n", args.steps, worstRho);
    std::printf("iterations: mean %.1f, max %lld\n",
                static_cast<double>(totalIters) / args.steps, maxIter);
    if (csv.is_open()) std::printf("wrote %s\n", args.csv.c_str());
    return 0;
}
