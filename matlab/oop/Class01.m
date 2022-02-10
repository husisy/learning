classdef Class01
    properties
        value
    end
    properties (Access=private)
        val1 = date()
    end

    methods
        function obj = class01(val)
            if nargin > 0
                if isnumeric(val)
                    obj.value = val;
                else
                    error('Value must be numeric')
                end
            end
        end

        function r = roundOff(obj)
            r = round([obj.value], 2);
        end

        function r = multiplyBy(obj,n)
            r = [obj.value] * n;
        end

        function ret = plus(obj1, obj2)
            ret = [obj1.value] + [obj2.value];
        end
    end

%     events (ListenAccess = protected)
%         StateChanged
%     end
end
