classdef State
    %HASHVEC Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        Value
    end
    
    methods
        function obj = State(value)
            %HASHVEC Construct an instance of this class
            %   Detailed explanation goes here
            obj.Value = value;
        end

        function obj = applyMove(obj, move)
            obj.Value = obj.Value(move);
        end
        
        function h = keyHash(obj)
            h = keyHash(obj.Value);
        end

        function tf = keyMatch(objA, objB)
            tf = isequal(objA.Value, objB.Value);
        end
    end
end

