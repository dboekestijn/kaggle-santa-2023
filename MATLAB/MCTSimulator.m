classdef MCTSimulator
    properties
        ExecutorPool
        MaxSimDepth
        SolutionState
        Moves
    end

    methods
        function obj = MCTSimulator(core_count, max_sim_depth, ...
                solution_state, moves)
            p = gcp('nocreate');
            if isempty(p)
                obj.ExecutorPool = parpool('Processes', core_count);
            else
                obj.ExecutorPool = p;
            end
            obj.MaxSimDepth = max_sim_depth;
            obj.SolutionState = mat2cell(solution_state, 1);
            obj.Moves = moves;
        end

        function shortest_move_path = simulateAll(obj, from_state, n)
            futures = cell(1, n);
            for i=1:n
                futures{i} = parfeval(...
                    obj.ExecutorPool, ...
                    @MCTSimulator.simulate, ...
                    1, ...
                    obj.MaxSimDepth, obj.SolutionState, obj.Moves, ...
                    mat2cell(from_state, 1) ...
                );
            end

            len_shortest_move_path = inf;
            shortest_move_path = [];
            for i=1:n
                move_path = fetchOutputs(futures{i});
                if isa(move_path, "logical")  % always false
                    continue
                end

                len_move_path = size(move_path, 1);
                if len_move_path < len_shortest_move_path
                    shortest_move_path = move_path;
                    len_shortest_move_path = len_move_path;
                end
            end
        end
    end

    methods (Static)
        function move_path = simulate(max_sim_depth, terminal_state, ...
                moves, from_state)
            visited_states = dictionary(from_state, 1);
            num_moves = size(moves, 1);
            move_path = zeros(max_sim_depth, 1);
            current_state = from_state;
            for sim_depth=1:max_sim_depth
                % idx = randi(num_moves, 1);
                % move = moves(idx, :);
                % move_path(sim_depth) = idx;
                % current_state = current_state(move);

                found_next_state = false;
                for idx=randperm(num_moves)
                    move = moves(idx, :);
                    next_state = mat2cell(current_state{1}(move), 1);
                    if ~isKey(visited_states, next_state)
                        visited_states = ...
                            insert(visited_states, next_state, 1);
                        move_path(sim_depth) = idx;
                        current_state = next_state;
                        found_next_state = true;
                    end
                end

                if ~found_next_state
                    if isequal(current_state{1}, from_state{1})
                        move_path = false;
                    end
                    return
                end

                if isequal(current_state{1}, terminal_state{1})
                    break
                end
            end

            move_path = move_path(1:sim_depth, :);
        end
    end

    methods
        function delete(obj)
            % cleanup when the MCTSimulator object is deleted
            delete(obj.ExecutorPool);
            delete(obj.MaxSimDepth);
            delete(obj.SolutionState);
            delete(gcp('nocreate'));
        end
    end

end
