// Distributed active-set QP solver.
//
//     min  sum_i  1/2 z_i' Q_i z_i + p_i' z_i
//     s.t. G_i z_i = g_i
//          H_i z_i <= h_i
//          sum_i A_i z_i <= b
//
// Local KKT systems are reduced to the null space of the active constraints.
// The central node handles the coupled constraints. Homotopy uses sigma from
// 1 to 0 so the warm start is exact at sigma = 1 and the original problem is
// recovered at sigma = 0.
//
// C++17 / Eigen 3. Port of the MATLAB/Python implementation.

#ifndef DMPC_HPP
#define DMPC_HPP

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace dmpc {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using Mask = std::vector<char>;

constexpr double EPS = std::numeric_limits<double>::epsilon();
constexpr double TINY = std::numeric_limits<double>::min();
constexpr double INF = std::numeric_limits<double>::infinity();

// utilities

inline std::vector<int> which(const Mask& m, bool val = true) {
    std::vector<int> idx;
    for (int i = 0; i < static_cast<int>(m.size()); ++i)
        if (static_cast<bool>(m[i]) == val) idx.push_back(i);
    return idx;
}

inline int count(const Mask& m) {
    return static_cast<int>(std::count(m.begin(), m.end(), 1));
}

inline bool any(const Mask& m) {
    return std::find(m.begin(), m.end(), 1) != m.end();
}

inline Mat rowsOf(const Mat& A, const std::vector<int>& idx) {
    Mat out(idx.size(), A.cols());
    for (size_t k = 0; k < idx.size(); ++k) out.row(k) = A.row(idx[k]);
    return out;
}

inline Vec pick(const Vec& v, const std::vector<int>& idx) {
    Vec out(idx.size());
    for (size_t k = 0; k < idx.size(); ++k) out(k) = v(idx[k]);
    return out;
}

// infinity norm; empty vectors return zero
inline double ninf(const Vec& v) { return v.size() ? v.cwiseAbs().maxCoeff() : 0.0; }

inline double norm1(const Mat& A) {
    return A.cols() ? A.cwiseAbs().colwise().sum().maxCoeff() : 0.0;
}

inline double spacing(double x) {
    return std::nextafter(std::abs(x), INF) - std::abs(x);
}

// Orthonormal basis of ker(A).
inline Mat nullSpace(const Mat& A, double rcond = -1.0) {
    if (A.rows() == 0) return Mat::Identity(A.cols(), A.cols());
    Eigen::JacobiSVD<Mat> svd(A, Eigen::ComputeFullV);
    const Vec& s = svd.singularValues();
    const double smax = s.size() ? s(0) : 0.0;
    const double rc = (rcond < 0.0)
                          ? EPS * std::max<Eigen::Index>(A.rows(), A.cols())
                          : rcond;
    int rank = 0;
    for (int i = 0; i < s.size(); ++i)
        if (s(i) > smax * rc) ++rank;
    return svd.matrixV().rightCols(A.cols() - rank);
}

// factorization

struct SeqQR {
    Mat V;                  // essential Householder vectors
    Vec beta;               // multipliers
    Mat T;                  // triangular factor
    Mask kept;              // columns that survived the dependence test
    int r = 0;              // rank
};

// Sequential Householder QR. Dependent columns are skipped in the given order.
inline SeqQR seqQrRankReveal(const Mat& A, double tol = -1.0) {
    const int n = static_cast<int>(A.rows());
    const int m = static_cast<int>(A.cols());
    if (tol < 0.0) tol = std::sqrt(EPS) * (m ? norm1(A) : 1.0);

    SeqQR out;
    out.V = Mat::Zero(n, std::max(m, 1));
    out.beta = Vec::Zero(std::max(m, 1));
    Mat R = Mat::Zero(std::max(n, 1), std::max(m, 1));
    out.kept.assign(m, 0);
    int k = 0;

    for (int i = 0; i < m; ++i) {
        Vec a = A.col(i);
        for (int j = 0; j < k; ++j) {
            const Vec v = out.V.col(j);
            a -= out.beta(j) * (v.dot(a)) * v;
        }
        if (a.tail(n - k).norm() <= tol) continue;   // dependent -> drop the row

        out.kept[i] = 1;
        Vec x = a.tail(n - k);
        const double s = (x(0) != 0.0) ? x(0) : 1.0;
        const double alpha = -std::copysign(x.norm(), s);
        Vec v = x;
        v(0) -= alpha;
        v /= v.norm();

        out.V.col(k).setZero();
        out.V.col(k).tail(n - k) = v;
        out.beta(k) = 2.0;
        if (k > 0) R.col(k).head(k) = a.head(k);
        R(k, k) = alpha;
        ++k;
    }

    out.V = out.V.leftCols(k).eval();
    out.beta = out.beta.head(k).eval();
    out.T = R.topLeftCorner(k, k).eval();
    out.r = k;
    return out;
}

// Reduced Hessian factorization.
class Reduced {
  public:
    void compute(const Mat& M) {
        n_ = static_cast<int>(M.rows());
        if (n_) ldlt_.compute(M);
    }
    Vec solve(const Vec& rhs) const {
        return n_ ? Vec(ldlt_.solve(rhs)) : Vec::Zero(0);
    }
    Mat solve(const Mat& rhs) const {
        return n_ ? Mat(ldlt_.solve(rhs)) : Mat::Zero(0, rhs.cols());
    }

  private:
    Eigen::LDLT<Mat> ldlt_;
    int n_ = 0;
};

// logging

// Count array elements sent between the nodes.
class Logger {
  public:
    struct Counts {
        long long send = 0, recv = 0;
    };

    explicit Logger(int nNodes = 0) { resize(nNodes); }

    void resize(int nNodes) { local_.assign(nNodes, Counts{}); }

    void reset() {
        central_ = Counts{};
        std::fill(local_.begin(), local_.end(), Counts{});
    }

    void centralSend(long long n) { central_.send += n; }
    void centralRecv(long long n) { central_.recv += n; }
    void localSend(int id, long long n) { local_[id].send += n; }
    void localRecv(int id, long long n) { local_[id].recv += n; }

    long long centralTotal() const { return central_.send + central_.recv; }
    long long localTotal() const {
        long long t = 0;
        for (const auto& c : local_) t += c.send + c.recv;
        return t;
    }
    const std::vector<Counts>& local() const { return local_; }

  private:
    Counts central_;
    std::vector<Counts> local_;
};

// local node

struct NodeData {
    Mat Q, G, H, A;
    Vec p, g, h, z0, locDual;
    Mask locActive;
};

class LocalNode {
  public:
    LocalNode(int id, const NodeData& d, const Vec& coupDual,
              const Mask& coupActive, Logger& log)
        : id_(id), log_(&log), B_(d.Q), G_(d.G), H_(d.H), A_(d.A),
          p_(d.p), g_(d.g), h_(d.h) {
        nx_ = static_cast<int>(B_.rows());
        neq_ = static_cast<int>(G_.rows());
        nineq_ = static_cast<int>(H_.rows());
        ncoup_ = static_cast<int>(A_.rows());

        Zg_ = neq_ ? nullSpace(G_) : Mat::Identity(nx_, nx_);
        Hz_ = H_ * Zg_;
        if (neq_) GGt_ = G_ * G_.transpose();

        x_ = d.z0;
        lam_ = d.locDual.head(neq_);
        mu_ = d.locDual.tail(nineq_);
        dlam_ = Vec::Zero(neq_);
        dmu_ = Vec::Zero(nineq_);

        actL_ = d.locActive;
        for (int i = 0; i < nineq_; ++i)
            if (!actL_[i]) mu_(i) = 0.0;

        nu_ = coupDual;
        dnu_ = Vec::Zero(ncoup_);
        actC_ = coupActive;
        for (int i = 0; i < ncoup_; ++i)
            if (!actC_[i]) nu_(i) = 0.0;

        initialize();
    }

    // accessors
    int nineq() const { return nineq_; }
    const Vec& x() const { return x_; }
    const Mask& activeL() const { return actL_; }
    double dtau() const { return dtau_; }

    Vec dual() const {
        Vec mu = mu_;
        for (int i = 0; i < nineq_; ++i)
            if (!actL_[i] || mu(i) < 0.0) mu(i) = 0.0;
        Vec out(neq_ + nineq_);
        out << lam_, mu;
        return out;
    }

    Vec residuals() const {
        const double rL =
            ninf(B_ * x_ + p_ + G_.transpose() * lam_ + H_.transpose() * mu_ +
                 A_.transpose() * nu_);
        const double rEq = neq_ ? ninf(G_ * x_ - g_) : 0.0;
        const double scH = std::max(1.0, ninf(h_));
        const double scM = std::max(1.0, ninf(mu_));
        const auto W = which(actL_);
        const double rTight =
            W.empty() ? 0.0 : ninf(rowsOf(H_, W) * x_ - pick(h_, W)) / scH;
        const double rViol = std::max(0.0, (H_ * x_ - h_).maxCoeff()) / scH;
        const double rComp =
            ninf((mu_.array() * (h_ - H_ * x_).array()).matrix()) / (scH * scM);
        Vec r(5);
        r << rL, rEq, rTight, rViol, rComp;
        return r;
    }

    // messaging

    void sendInitial(Vec& Ax, Mat& S, Vec& rho, bool& hasCoupled) {
        Ax = A_ * x_;
        log_->localSend(id_, Ax.size());
        hasCoupled = any(actC_);
        if (hasCoupled) {
            S = S_;
            rho = rho_;
            log_->localSend(id_, S_.size());
            log_->localSend(id_, rho_.size());
        }
    }

    void sendToCentral(Mat& S, Vec& rho, bool& stop) {
        S = S_;
        rho = rho_;
        stop = flag_;
        log_->localSend(id_, S_.size());
        log_->localSend(id_, rho_.size());
    }

    void sendChange(Mat& dS, Vec& drho, bool& stop) {
        dS = dS_;
        drho = drho_;
        stop = flag_;
        log_->localSend(id_, dS_.size());
        log_->localSend(id_, drho_.size());
    }

    void sendStep(Vec& Adx, double& dtau) {
        Adx = A_ * dx_;
        dtau = dtau_;
        log_->localSend(id_, Adx.size());
        log_->localSend(id_, 1);
    }

    void receiveDeltaNu(const Vec& dnuActive) {
        log_->localRecv(id_, dnuActive.size());
        const auto C = which(actC_);
        dnu_.setZero();
        for (size_t k = 0; k < C.size(); ++k) dnu_(C[k]) = dnuActive(k);

        sZ_ = sZbar_ - Mi_ * dnuActive;
        Vec s(nx_);
        s << sY_, sZ_;
        dx_ = Qi_ * s;
        YBZs_ = ZBY_.transpose() * sZ_;
        YAT_ = Y_.transpose() * rowsOf(A_, C).transpose();

        recoverMultipliers(Yrp_ - YBYs_ - YBZs_ - YAT_ * dnuActive);
        homotopyStep();
    }

    void dropCoupled(const std::vector<int>& dropped) {
        for (int d : dropped) {
            actC_[d] = 0;
            nu_(d) = 0.0;
            dnu_(d) = 0.0;
        }
        changeC_ = true;
        assemble();
    }

    // flag: 1 limiting, 0 not limiting, -1 terminal
    void receiveDeltaTau(double dtau, int flag, const Mask& actCnew) {
        if (flag == 1) {
            changeL_ = true;
            changeC_ = (actC_ != actCnew);
        } else if (flag == 0) {
            changeL_ = false;
            changeC_ = (actC_ != actCnew);
        }

        if (flag < 0) {                                  // terminal
            dtau_ = std::min(dtau, sigma_);
            advance();
            snapAndCheck(true);
            return;
        }

        log_->localRecv(id_, 2 + static_cast<long long>(actCnew.size()));

        dtau_ = std::min(dtau, sigma_);
        actC_ = actCnew;
        advance();

        double muS = std::max(ninf(mu_), 1.0);
        for (int i = 0; i < nineq_; ++i) {
            if (actL_[i] && std::abs(mu_(i)) <= 4 * EPS * muS) mu_(i) = 0.0;
            if (!actL_[i]) mu_(i) = 0.0;
        }

        if (flag == 1) {
            // These tests are independent; both may trigger.
            const bool didP = (dtau_ == dtauP_) && kP_ >= 0;
            const bool didD = (dtau_ == dtauD_) && kD_ >= 0;
            if (didP) activate(kP_);
            if (didD) {
                if (!didP) checkCurvature(kD_);
                actL_[kD_] = 0;
                mu_(kD_) = 0.0;
            }
        }

        snapAndCheck(false);
        updateTau();
    }

  private:
    // setup

    double rowScale(int i, const Vec& Hx) const {
        return std::max({std::abs(h_(i)), std::abs(Hx(i)), 1.0});
    }

    void initialize() {
        const Vec g0 = neq_ ? Vec(G_ * x_) : Vec::Zero(0);
        const Vec Hx = H_ * x_;

        Vec h0 = h_ + (Hx - h_).cwiseMax(0.0);
        for (int i = 0; i < nineq_; ++i)
            if (actL_[i]) h0(i) = Hx(i);
        for (int i = 0; i < nineq_; ++i)
            if (!actL_[i]) mu_(i) = 0.0;

        for (int i = 0; i < nineq_; ++i)
            if (!actL_[i] && Hx(i) - h0(i) > -1e-10)
                h0(i) = Hx(i) + slackRel_ * rowScale(i, Hx);

        buildGHA();
        SeqQR qr = seqQrRankReveal(GHA_);
        setQR(qr);

        if (std::count(qr.kept.begin(), qr.kept.end(), 0) > 0) {
            const std::vector<int> dropped = resolveDependency();
            buildGHA();
            SeqQR qr2 = seqQrRankReveal(GHA_);
            if (std::count(qr2.kept.begin(), qr2.kept.end(), 0) > 0)
                throw std::runtime_error("node " + std::to_string(id_) +
                                         ": rank deficient after drop");
            setQR(qr2);
            for (int d : dropped) h0(d) = Hx(d) + slackRel_ * rowScale(d, Hx);
        }

        p0_ = -(B_ * x_ + G_.transpose() * lam_ + H_.transpose() * mu_ +
                A_.transpose() * nu_);
        dp_ = p0_ - p_;
        dg_ = g0 - g_;
        dh_ = h0 - h_;

        pTau_ = p0_;
        gTau_ = g0;
        hTau_ = h0;
        flag_ = false;
        rp_ = dp_;
        rg_ = -dg_;

        factorLocal();
        if (any(actC_))
            coupledBlocks(true);
        else
            closeUncoupled();
    }

    void buildGHA() {
        const auto W = which(actL_);
        GHA_.resize(nx_, neq_ + static_cast<int>(W.size()));
        if (neq_) GHA_.leftCols(neq_) = G_.transpose();
        for (size_t k = 0; k < W.size(); ++k)
            GHA_.col(neq_ + k) = H_.row(W[k]).transpose();
    }

    void setQR(const SeqQR& qr) {
        V_ = qr.V;
        beta_ = qr.beta;
        T_ = qr.T;
        r_ = qr.r;
    }

    // Y/Z split and the uncoupled step.
    void factorLocal() {
        Qi_ = Mat::Identity(nx_, nx_);
        for (int j = r_ - 1; j >= 0; --j) {
            const Vec v = V_.col(j).tail(nx_ - j);
            Qi_.bottomRows(nx_ - j) -=
                beta_(j) * v * (v.transpose() * Qi_.bottomRows(nx_ - j));
        }
        Y_ = Qi_.leftCols(r_);
        Z_ = Qi_.rightCols(nx_ - r_);

        YBY_ = Y_.transpose() * B_ * Y_;
        ZBY_ = Z_.transpose() * B_ * Y_;
        ZBZ_ = Z_.transpose() * B_ * Z_;

        Yrp_ = Y_.transpose() * rp_;
        const Vec Zrp = Z_.transpose() * rp_;

        const auto W = which(actL_);
        Vec rgh(neq_ + static_cast<int>(W.size()));
        if (neq_) rgh.head(neq_) = rg_;
        for (size_t k = 0; k < W.size(); ++k) rgh(neq_ + k) = -dh_(W[k]);

        sY_ = r_ ? Vec(T_.transpose().triangularView<Eigen::Lower>().solve(rgh))
                 : Vec::Zero(0);

        YBYs_ = YBY_ * sY_;
        red_.compute(ZBZ_);
        ZBYs_ = ZBY_ * sY_;
        sZbar_ = red_.solve(Vec(Zrp - ZBYs_));

        Vec s(nx_);
        s << sY_, sZbar_;
        dxbar_ = Qi_ * s;
    }

    void coupledBlocks(bool force) {
        const auto C = which(actC_);
        const Mat Ac = rowsOf(A_, C);
        if (force || changeC_ || changeL_) ZAT_ = Z_.transpose() * Ac.transpose();
        Mi_ = red_.solve(ZAT_);
        const Mat S = ZAT_.transpose() * Mi_;
        const Vec rho = Ac * dxbar_;
        if (force || changeC_) {
            S_ = S;
            rho_ = rho;
        } else if (changeL_) {
            dS_ = S - S_;
            drho_ = rho - rho_;
            S_ = S;
            rho_ = rho;
        }
    }

    // No coupled rows: the local step is complete.
    void closeUncoupled() {
        sZ_ = sZbar_;
        dx_ = dxbar_;
        YBZs_ = ZBY_.transpose() * sZ_;
        recoverMultipliers(Yrp_ - YBYs_ - YBZs_);
        homotopyStep();
    }

    void recoverMultipliers(const Vec& rhs) {
        const Vec dlm =
            r_ ? Vec(T_.triangularView<Eigen::Upper>().solve(rhs)) : Vec::Zero(0);
        dlam_ = dlm.head(neq_);
        dmu_.setZero();
        const auto W = which(actL_);
        for (size_t k = 0; k < W.size(); ++k) dmu_(W[k]) = dlm(neq_ + k);
    }

    void assemble() {
        if (changeL_) {
            buildGHA();
            SeqQR qr = seqQrRankReveal(GHA_);
            setQR(qr);

            if (r_ < std::min(GHA_.rows(), GHA_.cols())) {
                for (int i = 0; i < neq_; ++i)
                    if (!qr.kept[i])
                        throw std::runtime_error("node " + std::to_string(id_) +
                                                 ": equality block rank deficient");
                const std::vector<int> dropped = resolveDependency();
                for (int d : dropped) dmu_(d) = 0.0;
                buildGHA();
                SeqQR qr2 = seqQrRankReveal(GHA_);
                setQR(qr2);
                if (r_ < std::min(GHA_.rows(), GHA_.cols()))
                    throw std::runtime_error("node " + std::to_string(id_) +
                                             ": still rank deficient");
            }
            factorLocal();
        }

        if (any(actC_))
            coupledBlocks(false);
        else
            closeUncoupled();
    }

    // dependency handling

    static void dualRatio(const Vec& muW, const Vec& c, double& t, int& j) {
        t = INF;
        j = -1;
        for (int i = 0; i < c.size(); ++i) {
            if (c(i) <= 0.0) continue;
            const double ratio = muW(i) / c(i);
            if (ratio < t) {
                t = ratio;
                j = i;
            }
        }
    }

    // Drop dependent active rows. The leaving row is chosen by the dual ratio test.
    std::vector<int> resolveDependency() {
        const Vec mu0 = mu_;
        const Vec lam0 = lam_;
        const double tol = 1e-10;
        std::vector<int> dropped;

        Vec nrm = Hz_.rowwise().norm();
        const double nmax = std::max(nrm.size() ? nrm.maxCoeff() : 0.0, TINY);
        for (int i = 0; i < nineq_; ++i) {
            if (actL_[i] && nrm(i) <= tol * nmax) {
                mu_(i) = 0.0;
                actL_[i] = 0;
                dropped.push_back(i);
            }
        }

        while (true) {
            const auto W = which(actL_);
            if (W.size() <= 1) break;

            const Mat Hw = rowsOf(Hz_, W);
            Vec s = Hw.rowwise().norm().cwiseMax(TINY);
            const Mat D = nullSpace((s.asDiagonal().inverse() * Hw).transpose(), tol);
            if (D.cols() == 0) break;

            Vec c = D.col(0).cwiseQuotient(s);
            Vec muW(W.size());
            for (size_t k = 0; k < W.size(); ++k) muW(k) = mu_(W[k]);

            double tp, tm;
            int jp, jm;
            dualRatio(muW, c, tp, jp);
            dualRatio(muW, Vec(-c), tm, jm);
            double t;
            int j;
            if (tp <= tm) {
                t = tp;
                j = jp;
            } else {
                t = tm;
                j = jm;
                c = -c;
            }
            if (!std::isfinite(t)) break;

            Vec upd = muW - t * c;
            upd(j) = 0.0;
            for (size_t k = 0; k < W.size(); ++k) mu_(W[k]) = std::max(upd(k), 0.0);
            actL_[W[j]] = 0;
            dropped.push_back(W[j]);
        }

        if (!dropped.empty() && neq_) {
            // This compensation is exact.
            const Vec r = H_.transpose() * (mu_ - mu0);
            lam_ -= GGt_.ldlt().solve(G_ * r);
            const double jump = ninf(H_.transpose() * (mu_ - mu0) +
                                     G_.transpose() * (lam_ - lam0));
            if (jump > 1e-8 * std::max(ninf(H_.transpose() * mu0), 1.0))
                throw std::runtime_error("node " + std::to_string(id_) +
                                         ": drop left a stationarity jump");
        }
        return dropped;
    }

    // ratio tests

    void homotopyStep() {
        const auto W = which(actL_);
        double muS = 1.0, dmuS = TINY;
        for (int i : W) {
            muS = std::max(muS, std::abs(mu_(i)));
            dmuS = std::max(dmuS, std::abs(dmu_(i)));
        }

        for (int i : W)
            if (std::abs(mu_(i)) <= 4 * EPS * muS) mu_(i) = 0.0;

        dtauD_ = INF;
        kD_ = -1;
        for (int i : W) {
            if (dmu_(i) >= -1e-10 * dmuS) continue;
            const double ratio = std::max(-mu_(i) / dmu_(i), 0.0);
            if (ratio < dtauD_) {
                dtauD_ = ratio;
                kD_ = i;
            }
        }

        const auto Cn = which(actL_, false);
        const Mat Hc = rowsOf(H_, Cn);
        const Vec Hdx = Hc * dx_;
        const Vec num = (pick(hTau_, Cn) - Hc * x_).cwiseMax(0.0);
        const Vec den = Hdx + pick(dh_, Cn);
        const double denS = std::max({ninf(Hdx) + ninf(dh_), ninf(h_), 1.0});

        dtauP_ = INF;
        kP_ = -1;
        for (int k = 0; k < static_cast<int>(Cn.size()); ++k) {
            if (den(k) <= 1e-10 * denS) continue;
            const double ratio = num(k) / den(k);
            if (ratio < dtauP_) {
                dtauP_ = ratio;
                kP_ = Cn[k];
            }
        }

        dtau_ = std::min(dtauP_, dtauD_);
        if (!(dtau_ >= 0.0))
            throw std::runtime_error("node " + std::to_string(id_) + ": negative step");
    }

    // stepping

    Vec applyReflectors(const Vec& a0, bool forward) const {
        Vec a = a0;
        for (int t = 0; t < r_; ++t) {
            const int j = forward ? t : r_ - 1 - t;
            const Vec v = V_.col(j).tail(nx_ - j);
            a.tail(nx_ - j) -= beta_(j) * v * (v.dot(a.tail(nx_ - j)));
        }
        return a;
    }

    void advance() {
        x_ += dtau_ * dx_;
        lam_ += dtau_ * dlam_;
        mu_ += dtau_ * dmu_;
        nu_ += dtau_ * dnu_;
        sigma_ -= dtau_;
        tau_ = 1.0 - sigma_;
    }

    void snapAndCheck(bool terminal) {
        double muS = std::max(ninf(mu_), 1.0);
        for (int i = 0; i < nineq_; ++i) {
            if (actL_[i] && std::abs(mu_(i)) <= 4 * EPS * muS) mu_(i) = 0.0;
            if (!actL_[i]) mu_(i) = 0.0;
        }
        muS = std::max(ninf(mu_), 1.0);
        if (mu_.size() && mu_.minCoeff() < -1e-10 * muS)
            throw std::runtime_error(
                "node " + std::to_string(id_) +
                (terminal ? ": terminal step drove mu negative; the path was "
                            "closed past a dual blocking point"
                          : ": genuine negative multiplier"));
        mu_ = mu_.cwiseMax(0.0);
    }

    void updateTau() {
        // Data is affine in sigma; sigma = 0 gives the original problem.
        pTau_ = p_ + sigma_ * dp_;
        gTau_ = g_ + sigma_ * dg_;
        hTau_ = h_ + sigma_ * dh_;
        flag_ = sigma_ < 1e-14;
        assemble();
    }

    // Add row k, exchanging a blocking row if necessary.
    void activate(int k) {
        const Vec QTH = applyReflectors(Vec(H_.row(k).transpose()), true);
        const Vec YH = QTH.head(r_);
        const Vec ZH = QTH.tail(nx_ - r_);

        const Vec py = Vec::Zero(r_);
        const Vec pz = red_.solve(ZH);
        Vec pcat(nx_);
        pcat << py, pz;
        const Vec pdir = applyReflectors(pcat, false);

        const Vec rhs = YH - YBY_ * py - ZBY_.transpose() * pz;
        const Vec zx =
            r_ ? Vec(T_.triangularView<Eigen::Upper>().solve(rhs)) : Vec::Zero(0);
        const Vec zeta = zx.head(neq_);
        Vec xi = Vec::Zero(nineq_);
        const auto W = which(actL_);
        for (size_t t = 0; t < W.size(); ++t) xi(W[t]) = zx(neq_ + t);

        const double tolDep =
            std::sqrt(EPS) * std::max(norm1(GHA_), H_.row(k).cwiseAbs().sum());

        if (ZH.norm() <= tolDep) {
            double theta = INF;
            int leaving = -1;
            for (int i = 0; i < nineq_; ++i) {
                if (!actL_[i] || xi(i) <= 0.0) continue;
                const double ratio = mu_(i) / xi(i);
                if (ratio < theta) {
                    theta = ratio;
                    leaving = i;
                }
            }
            if (leaving < 0)
                throw std::runtime_error("node " + std::to_string(id_) +
                                         ": QP appears infeasible");
            theta = std::max(theta, 0.0);
            lam_ -= theta * zeta;
            mu_ -= theta * xi;
            actL_[leaving] = 0;
            mu_(leaving) = 0.0;
            mu_(k) = theta;
        }
        actL_[k] = 1;
    }

    // Zero curvature after release means an unbounded direction.
    void checkCurvature(int k) {
        const auto W = which(actL_);
        Vec eA = Vec::Zero(W.size());
        for (size_t t = 0; t < W.size(); ++t)
            if (W[t] == k) eA(t) = 1.0;

        Vec rhsP(neq_ + static_cast<int>(W.size()));
        rhsP << Vec::Zero(neq_), -eA;
        const Vec py =
            r_ ? Vec(T_.transpose().triangularView<Eigen::Lower>().solve(rhsP))
               : Vec::Zero(0);
        const Vec pz = red_.solve(Vec(-ZBY_ * py));
        Vec pcat(nx_);
        pcat << py, pz;
        const Vec pdir = applyReflectors(pcat, false);

        const Vec rhs = -YBY_ * py - ZBY_.transpose() * pz;
        const Vec zx =
            r_ ? Vec(T_.triangularView<Eigen::Upper>().solve(rhs)) : Vec::Zero(0);
        if (ninf(zx) >= 1e-14 * std::max(ninf(mu_), 1.0)) return;

        const Vec hTau = h_ + sigma_ * dh_;
        const auto F = which(actL_, false);
        const Mat Hf = rowsOf(H_, F);
        const Vec num = pick(hTau, F) - Hf * x_;
        const Vec den = Hf * pdir;
        const double denS = std::max(ninf(den), TINY);

        double best = INF;
        int bestRow = -1;
        for (int t = 0; t < static_cast<int>(F.size()); ++t) {
            if (den(t) <= 1e-10 * denS) continue;
            const double ratio = std::max(num(t) / den(t), 0.0);
            if (ratio < best) {
                best = ratio;
                bestRow = F[t];
            }
        }
        if (bestRow < 0)
            throw std::runtime_error("node " + std::to_string(id_) +
                                     ": unbounded along the released direction");
        if (best > 1e10)
            throw std::runtime_error("node " + std::to_string(id_) +
                                     ": QP appears infeasible");
        x_ += best * pdir;
        actL_[bestRow] = 1;
    }

    // state
    int id_;
    Logger* log_;
    Mat B_, G_, H_, A_;
    Vec p_, g_, h_;
    int nx_ = 0, neq_ = 0, nineq_ = 0, ncoup_ = 0;

    Mat Zg_, Hz_, GGt_;
    Vec x_, lam_, mu_, dlam_, dmu_, nu_, dnu_, dx_, dxbar_;
    Mask actL_, actC_;

    Vec p0_, dp_, dg_, dh_, pTau_, gTau_, hTau_, rp_, rg_;
    Mat GHA_, V_, Qi_, Y_, Z_, YBY_, ZBY_, ZBZ_, ZAT_, Mi_, YAT_;
    Vec beta_, Yrp_, sY_, YBYs_, ZBYs_, sZbar_, sZ_, YBZs_;
    Mat T_;
    Reduced red_;
    int r_ = 0;

    Mat S_, dS_;
    Vec rho_, drho_;

    double sigma_ = 1.0, tau_ = 0.0, dtau_ = 0.0;
    double dtauP_ = INF, dtauD_ = INF;
    int kP_ = -1, kD_ = -1;
    bool flag_ = false, changeL_ = false, changeC_ = false;

    // Tighter tolerance for the initial active-set check.
    double slackRel_ = 1e-6;
};

// central node

class CentralNode {
  public:
    CentralNode(std::vector<LocalNode>& nodes, const Vec& b, const Vec& coupDual,
                const Mask& coupActive, Logger& log)
        : nodes_(&nodes), log_(&log), b_(b) {
        n_ = static_cast<int>(b_.size());
        nu_ = coupDual;
        dnu_ = Vec::Zero(n_);
        actC_ = coupActive;
        dtaus_.assign(nodes.size() + 1, 0.0);
        changeC_ = any(actC_);

        AxSum_ = Vec::Zero(n_);
        AdxSum_ = Vec::Zero(n_);
        const int nA = count(actC_);
        Ssum_ = Mat::Zero(nA, nA);
        rhoSum_ = Vec::Zero(nA);

        // Prevent an endless sequence of zero-length steps.
        long long rows = n_;
        for (auto& nd : nodes) rows += nd.nineq();
        zeroBudget_ = 2 * rows;

        initialize();
    }

    const Mask& activeC() const { return actC_; }
    double sigma() const { return sigma_; }
    int limitingAgent() const { return id_; }

    Vec dual() const {
        Vec nu = nu_;
        for (int i = 0; i < n_; ++i)
            if (!actC_[i] || nu(i) < 0.0) nu(i) = 0.0;
        return nu;
    }

    Vec residuals() const {
        const auto C = which(actC_);
        const double r = C.empty() ? 0.0 : ninf(pick(AxSum_, C) - pick(b_, C));
        Vec out(2);
        out << r, std::abs(nu_.dot(b_ - AxSum_));
        return out;
    }

    void aggregateAndSolve() {
        for (size_t pass = 0; pass <= nodes_->size(); ++pass) {
            if (!any(actC_)) {
                dnu_.setZero();
                break;
            }

            if (changeC_) {
                const int nA = count(actC_);
                Ssum_ = Mat::Zero(nA, nA);
                rhoSum_ = Vec::Zero(nA);
                for (auto& nd : *nodes_) {
                    Mat S;
                    Vec rho;
                    bool stop;
                    nd.sendToCentral(S, rho, stop);
                    log_->centralRecv(S.size());
                    log_->centralRecv(rho.size());
                    Ssum_ += S;
                    rhoSum_ += rho;
                }
            } else {
                Mat dS;
                Vec drho;
                bool stop;
                (*nodes_)[id_].sendChange(dS, drho, stop);
                log_->centralRecv(dS.size());
                log_->centralRecv(drho.size());
                Ssum_ += dS;
                rhoSum_ += drho;
            }

            Eigen::ColPivHouseholderQR<Mat> qr(Ssum_);
            const Vec rdiag = qr.matrixR().diagonal().cwiseAbs();
            const double tol =
                std::max(Ssum_.rows(), Ssum_.cols()) * spacing(rdiag(0));
            int r = 0;
            for (int i = 0; i < rdiag.size(); ++i)
                if (rdiag(i) > tol) ++r;

            if (r < Ssum_.cols()) {
                const auto active = which(actC_);
                const auto& perm = qr.colsPermutation().indices();
                std::vector<int> drop;
                for (int i = r; i < perm.size(); ++i) drop.push_back(active[perm(i)]);
                dropRows(drop);
                continue;
            }

            const auto C = which(actC_);
            dnuActive_ = Ssum_.lu().solve(Vec(rhoSum_ - pick(rb_, C)));
            dnu_.setZero();
            for (size_t k = 0; k < C.size(); ++k) dnu_(C[k]) = dnuActive_(k);
            for (auto& nd : *nodes_) {
                log_->centralSend(dnuActive_.size());
                nd.receiveDeltaNu(dnuActive_);
            }
            break;
        }

        AdxSum_.setZero();
        for (size_t i = 0; i < nodes_->size(); ++i) {
            Vec Adx;
            double dt;
            (*nodes_)[i].sendStep(Adx, dt);
            log_->centralRecv(Adx.size());
            log_->centralRecv(1);
            AdxSum_ += Adx;
            dtaus_[i] = dt;
        }
    }

    // returns true when the homotopy has reached sigma = 0
    bool homotopyStep() {
        // dual ratio test
        const auto C = which(actC_);
        const double dnuS = std::max(ninf(pick(dnu_, C)), TINY);
        double dtauD = INF;
        int ka = -1;
        for (int i : C) {
            if (dnu_(i) >= -1e-10 * dnuS) continue;
            const double ratio = std::max(-nu_(i) / dnu_(i), 0.0);
            if (ratio < dtauD) {
                dtauD = ratio;
                ka = i;
            }
        }

        // primal ratio test
        const auto Cn = which(actC_, false);
        const double denS = std::max(ninf(AdxSum_) + ninf(rb_), TINY);
        double dtauP = INF;
        int l = -1;
        for (int i : Cn) {
            const double den = AdxSum_(i) - rb_(i);
            if (den <= 1e-10 * denS) continue;
            const double num = std::max(bTau_(i) - AxSum_(i), 0.0);
            const double ratio = num / den;
            if (ratio < dtauP) {
                dtauP = ratio;
                l = i;
            }
        }

        dtaus_.back() = std::min(dtauP, dtauD);
        id_ = static_cast<int>(std::min_element(dtaus_.begin(), dtaus_.end()) -
                               dtaus_.begin());
        const double dtauC = dtaus_[id_];
        if (dtauC < 0.0) throw std::runtime_error("central: negative step");
        dtau_ = dtauC;

        if (std::min(dtau_, sigma_) <= 0.0)
            ++nZero_;
        else
            nZero_ = 0;

        double step;
        if (sigma_ <= sigmaTol_) {
            step = sigma_;
        } else if (nZero_ >= zeroBudget_) {
            throw std::runtime_error("degenerate cycle: " + std::to_string(nZero_) +
                                     " zero-length steps");
        } else {
            step = std::min(dtau_, sigma_);
        }

        AxSum_ += step * AdxSum_;
        nu_ += step * dnu_;
        const double nuS = std::max(ninf(nu_), 1.0);
        for (int i : which(actC_))
            if (std::abs(nu_(i)) <= 4 * EPS * nuS) nu_(i) = 0.0;

        sigma_ -= step;
        tau_ = 1.0 - sigma_;

        if (sigma_ < 1e-14) {
            for (auto& nd : *nodes_) nd.receiveDeltaTau(step, -1, actC_);
            return true;
        }

        const double tolTie = 1e-12 * std::max(sigma_, 1.0);
        std::vector<int> localActs;
        for (size_t i = 0; i + 1 < dtaus_.size(); ++i)
            if (std::isfinite(dtaus_[i]) && dtaus_[i] <= dtauC + tolTie)
                localActs.push_back(static_cast<int>(i));
        const bool centralTied =
            std::isfinite(dtaus_.back()) && dtaus_.back() <= dtauC + tolTie;

        if (centralTied) {
            changeC_ = true;
            if (dtauP <= dtauC + tolTie && l >= 0) actC_[l] = 1;
            if (dtauD <= dtauC + tolTie && ka >= 0) {
                actC_[ka] = 0;
                nu_(ka) = 0.0;
            }
        } else {
            // Multiple local changes invalidate the incremental update.
            changeC_ = localActs.size() > 1;
        }

        if (changeC_) {
            const int nA = count(actC_);
            Ssum_ = Mat::Zero(nA, nA);
            rhoSum_ = Vec::Zero(nA);
        }

        for (size_t i = 0; i < nodes_->size(); ++i) {
            const bool acting =
                std::find(localActs.begin(), localActs.end(),
                          static_cast<int>(i)) != localActs.end();
            log_->centralSend(2 + static_cast<long long>(actC_.size()));
            (*nodes_)[i].receiveDeltaTau(dtau_, acting ? 1 : 0, actC_);
        }

        bTau_ = b_ - sigma_ * rb_;      // == b bitwise at sigma == 0
        return false;
    }

  private:
    void initialize() {
        for (auto& nd : *nodes_) {
            Vec Ax, rho;
            Mat S;
            bool hasCoupled;
            nd.sendInitial(Ax, S, rho, hasCoupled);
            log_->centralRecv(Ax.size());
            AxSum_ += Ax;
            if (hasCoupled) {
                log_->centralRecv(S.size());
                log_->centralRecv(rho.size());
                Ssum_ += S;
                rhoSum_ += rho;
            }
        }

        const Vec viol = AxSum_ - b_;
        const double nuS = std::max(ninf(nu_), 1.0);
        for (int i = 0; i < n_; ++i) {
            const double sc = std::max(std::abs(viol(i)), 1.0);
            if (actC_[i] && nu_(i) <= 1e-12 * nuS && viol(i) < -1e-10 * sc) {
                actC_[i] = 0;
                nu_(i) = 0.0;
            }
        }

        b0_ = b_ + viol.cwiseMax(0.0);
        for (int i = 0; i < n_; ++i) {
            if (actC_[i]) b0_(i) = AxSum_(i);
            if (!actC_[i]) nu_(i) = 0.0;
        }
        for (int i = 0; i < n_; ++i) {
            if (!actC_[i] && AxSum_(i) - b0_(i) > -1e-10) {
                const double s =
                    std::max({std::abs(b_(i)), std::abs(AxSum_(i)), 1.0});
                b0_(i) = AxSum_(i) + slackRel_ * s;
            }
        }

        rb_ = b_ - b0_;
        bTau_ = b0_;
    }

    void dropRows(const std::vector<int>& dropped) {
        for (int d : dropped) {
            actC_[d] = 0;
            nu_(d) = 0.0;
        }
        for (auto& nd : *nodes_) nd.dropCoupled(dropped);
        changeC_ = true;
        const int nA = count(actC_);
        Ssum_ = Mat::Zero(nA, nA);
        rhoSum_ = Vec::Zero(nA);
    }

    std::vector<LocalNode>* nodes_;
    Logger* log_;
    Vec b_, nu_, dnu_, dnuActive_, AxSum_, AdxSum_, b0_, rb_, bTau_;
    Mat Ssum_;
    Vec rhoSum_;
    Mask actC_;
    std::vector<double> dtaus_;
    int n_ = 0, id_ = 0;
    long long nZero_ = 0, zeroBudget_ = 0;
    double sigma_ = 1.0, tau_ = 0.0, dtau_ = 0.0;
    double sigmaTol_ = 1e-14, slackRel_ = 1e-6;
    bool changeC_ = false;
};

// driver

struct Problem {
    std::vector<NodeData> nodes;
    Vec b, coupDual;
    Mask coupActive;
};

struct Solution {
    bool converged = false;
    int iter = 0;
    double sigma = 1.0;
    std::vector<Vec> z, dualL, res;
    Vec dualC;
    std::vector<Mask> activeL;
    Mask activeC;
    std::vector<int> nAc, limiting;
};

inline Solution solve(const Problem& prob, Logger& log, int maxIters = 10000) {
    std::vector<LocalNode> nodes;
    nodes.reserve(prob.nodes.size());
    for (size_t i = 0; i < prob.nodes.size(); ++i)
        nodes.emplace_back(static_cast<int>(i), prob.nodes[i], prob.coupDual,
                           prob.coupActive, log);

    CentralNode central(nodes, prob.b, prob.coupDual, prob.coupActive, log);

    Solution sol;
    for (int it = 1; it <= maxIters; ++it) {
        central.aggregateAndSolve();
        const bool done = central.homotopyStep();
        sol.iter = it;
        sol.nAc.push_back(count(central.activeC()));
        sol.limiting.push_back(central.limitingAgent());
        if (done) {
            sol.converged = true;
            break;
        }
    }

    sol.sigma = central.sigma();
    sol.dualC = central.dual();
    sol.activeC = central.activeC();
    for (auto& nd : nodes) {
        sol.z.push_back(nd.x());
        sol.dualL.push_back(nd.dual());
        sol.activeL.push_back(nd.activeL());
        sol.res.push_back(nd.residuals());
    }
    sol.res.push_back(central.residuals());

    // Do not return a non-converged point.
    if (!sol.converged) sol.z.clear();
    return sol;
}

// Relative KKT residual.
inline double kktResidual(const Problem& prob, const Solution& sol) {
    const int M = static_cast<int>(prob.nodes.size());
    const int n = static_cast<int>(prob.nodes[0].Q.rows());
    const int ne = static_cast<int>(prob.nodes[0].G.rows());
    const int ni = static_cast<int>(prob.nodes[0].H.rows());
    const int nc = static_cast<int>(prob.b.size());

    Vec z(M * n), p(M * n), g(M * ne), h(M * ni), lam(M * ne), mu(M * ni);
    Mat A(nc, M * n);
    Vec Gz(M * ne), Hz(M * ni), stat = Vec::Zero(M * n);
    Vec Az = Vec::Zero(nc);

    for (int i = 0; i < M; ++i) {
        const NodeData& d = prob.nodes[i];
        z.segment(i * n, n) = sol.z[i];
        p.segment(i * n, n) = d.p;
        g.segment(i * ne, ne) = d.g;
        h.segment(i * ni, ni) = d.h;
        lam.segment(i * ne, ne) = sol.dualL[i].head(ne);
        mu.segment(i * ni, ni) = sol.dualL[i].tail(ni);
        A.middleCols(i * n, n) = d.A;

        Gz.segment(i * ne, ne) = d.G * sol.z[i] - d.g;
        Hz.segment(i * ni, ni) = d.H * sol.z[i] - d.h;
        stat.segment(i * n, n) = d.Q * sol.z[i] + d.p +
                                 d.G.transpose() * sol.dualL[i].head(ne) +
                                 d.H.transpose() * sol.dualL[i].tail(ni) +
                                 d.A.transpose() * sol.dualC;
        Az += d.A * sol.z[i];
    }

    Vec y(M * ni + nc), r(M * ni + nc);
    y << mu, sol.dualC;
    r << Hz, Az - prob.b;

    const double sp = std::max({1.0, ninf(h), ninf(prob.b)});
    const double sd = std::max(1.0, ninf(y));

    return std::max({ninf(stat) / std::max(1.0, ninf(p)),
                     ninf(Gz) / std::max(1.0, ninf(g)),
                     std::max(0.0, r.maxCoeff()) / sp,
                     std::max(0.0, -y.minCoeff()) / sd,
                     ninf((y.array() * r.array()).matrix()) / (sp * sd)});
}

}  // namespace dmpc

#endif  // DMPC_HPP
