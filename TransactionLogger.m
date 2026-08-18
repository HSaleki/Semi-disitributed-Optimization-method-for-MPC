classdef TransactionLogger < handle
    properties
        % Central node separate counters:
        centralSendDoubleCount = 0;
        centralSendIntegerCount = 0;
        centralReceiveDoubleCount = 0;
        centralReceiveIntegerCount = 0;
        
        % Local node overall counters (across all nodes):
        localSendDoubleCount = 0;
        localSendIntegerCount = 0;
        localReceiveDoubleCount = 0;
        localReceiveIntegerCount = 0;
        
        % A containers.Map to hold per-local-node transfer counts.
        % Keys are node IDs (as strings) and values are structs with fields:
        %   send: struct('doubleCount',0, 'integerCount',0)
        %   receive: struct('doubleCount',0, 'integerCount',0)
        localTransfers;
    end
    
    methods
        function obj = TransactionLogger()
            % Initialize the containers.Map with char keys.
            obj.localTransfers = containers.Map('KeyType','char','ValueType','any');
        end
        
        function logData(obj, origin, data, tag, varargin)
            % logData now requires a tag (e.g., 'send' or 'receive').
            % For a local origin, a node ID must also be provided.
            %
            % Usage:
            %   For central node:
            %       TransactionLogger.getInstance().logData('central', data, 'send');
            %       TransactionLogger.getInstance().logData('central', data, 'receive');
            %
            %   For local node:
            %       TransactionLogger.getInstance().logData('local', data, 'send', nodeId);
            %       TransactionLogger.getInstance().logData('local', data, 'receive', nodeId);
            
            % Verify tag validity:
            if ~ismember(tag, {'send','receive'})
                error('Tag must be either ''send'' or ''receive''.');
            end
            
            if ~isnumeric(data)
                %fprintf('Data not numeric, skipped logging. Origin: %s, Tag: %s\n', origin, tag);
                return;
            end
            
            numElements = numel(data);
            % For doubles, count every element as a double.
            if isa(data, 'double')
                numDoubles = numElements;
                numIntegers = 0;
            elseif any(strcmp(class(data), {'int8','uint8','int16','uint16','int32','uint32','int64','uint64'}))
                numIntegers = numElements;
                numDoubles = 0;
            else
                numIntegers = 0;
                numDoubles = 0;
            end
            
            if strcmp(origin, 'central')
                if strcmp(tag, 'send')
                    obj.centralSendDoubleCount = obj.centralSendDoubleCount + numDoubles;
                    obj.centralSendIntegerCount = obj.centralSendIntegerCount + numIntegers;
                    %fprintf('Central logged (send): %d doubles, %d integers.\n', numDoubles, numIntegers);
                elseif strcmp(tag, 'receive')
                    obj.centralReceiveDoubleCount = obj.centralReceiveDoubleCount + numDoubles;
                    obj.centralReceiveIntegerCount = obj.centralReceiveIntegerCount + numIntegers;
                    %fprintf('Central logged (receive): %d doubles, %d integers.\n', numDoubles, numIntegers);
                end
            elseif strcmp(origin, 'local')
                if isempty(varargin)
                    error('For local origin, a node ID must be provided.');
                end
                nodeId = num2str(varargin{1});
                if strcmp(tag, 'send')
                    obj.localSendDoubleCount = obj.localSendDoubleCount + numDoubles;
                    obj.localSendIntegerCount = obj.localSendIntegerCount + numIntegers;
                    %fprintf('Local logged (global, send): %d doubles, %d integers.\n', numDoubles, numIntegers);
                elseif strcmp(tag, 'receive')
                    obj.localReceiveDoubleCount = obj.localReceiveDoubleCount + numDoubles;
                    obj.localReceiveIntegerCount = obj.localReceiveIntegerCount + numIntegers;
                    %fprintf('Local logged (global, receive): %d doubles, %d integers.\n', numDoubles, numIntegers);
                end
                
                % Update per-node counts:
                if ~obj.localTransfers.isKey(nodeId)
                    % Initialize structure for this node.
                    initialCounts = struct('send', struct('doubleCount',0, 'integerCount',0), ...
                                             'receive', struct('doubleCount',0, 'integerCount',0));
                    obj.localTransfers(nodeId) = initialCounts;
                end
                nodeStruct = obj.localTransfers(nodeId);
                if strcmp(tag, 'send')
                    nodeStruct.send.doubleCount = nodeStruct.send.doubleCount + numDoubles;
                    nodeStruct.send.integerCount = nodeStruct.send.integerCount + numIntegers;
                    %fprintf('Local node %s logged (send): %d doubles, %d integers.\n', nodeId, numDoubles, numIntegers);
                elseif strcmp(tag, 'receive')
                    nodeStruct.receive.doubleCount = nodeStruct.receive.doubleCount + numDoubles;
                    nodeStruct.receive.integerCount = nodeStruct.receive.integerCount + numIntegers;
                    %fprintf('Local node %s logged (receive): %d doubles, %d integers.\n', nodeId, numDoubles, numIntegers);
                end
                obj.localTransfers(nodeId) = nodeStruct; % Save update.
            else
                %fprintf('Unknown origin: %s\n', origin);
            end
        end
        
        function counts = getCounts(obj)
            % Returns a structure containing the counters.
            counts.central.send = struct('doubleCount', obj.centralSendDoubleCount, ...
                                           'integerCount', obj.centralSendIntegerCount);
            counts.central.receive = struct('doubleCount', obj.centralReceiveDoubleCount, ...
                                              'integerCount', obj.centralReceiveIntegerCount);
            counts.local.send = struct('doubleCount', obj.localSendDoubleCount, ...
                                        'integerCount', obj.localSendIntegerCount);
            counts.local.receive = struct('doubleCount', obj.localReceiveDoubleCount, ...
                                           'integerCount', obj.localReceiveIntegerCount);
            counts.localBreakdown = obj.localTransfers;
        end
        
        function resetCounts(obj)
            % Reset all counters.
            obj.centralSendDoubleCount = 0;
            obj.centralSendIntegerCount = 0;
            obj.centralReceiveDoubleCount = 0;
            obj.centralReceiveIntegerCount = 0;
            obj.localSendDoubleCount = 0;
            obj.localSendIntegerCount = 0;
            obj.localReceiveDoubleCount = 0;
            obj.localReceiveIntegerCount = 0;
            obj.localTransfers = containers.Map('KeyType','char','ValueType','any');
            %fprintf('Logger counts have been reset.\n');
        end
    end
    
    methods (Static)
        function logger = getInstance()
            persistent uniqueInstance;
            if isempty(uniqueInstance) || ~isvalid(uniqueInstance)
                uniqueInstance = TransactionLogger();
            end
            logger = uniqueInstance;
        end
    end
end
