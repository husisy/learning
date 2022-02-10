# MATLAB OOP

1. link
   * [official documentation](https://www.mathworks.com/help/matlab/object-oriented-programming.html)
2. Objects Manage Internal State
   * Constrain the data values assigned to any given property
   * Calculate the value of a property only when it is queried
   * Broadcast notices when any property value is queried or changed
   * Restrict access to properties and methods
3. Reducing Redundancy
   * Check inputs
   * Perform computation on the first input argument
   * Transform the result of step 2 based on the second input argument
   * Check validity of outputs and return values
4. Defining Consistent Interfaces
   * Identify the requirements of a particular objective
   * Encode requirements into your program as an interface class
5. Reducing Complexity
   * Objects provide an interface that hides implementation details.
   * Objects enforce rules that control how objects interact.
6. property: `public`, `protected`, `private`
7. key terms
   * Class definition — Description of what is common to every instance of a class.
   * Properties — Data storage for class instances
   * Methods — Special functions that implement operations that are usually performed only on instances of the class
   * Events — Messages defined by classes and broadcast by class instances when some specific action occurs
   * Attributes — Values that modify the behavior of properties, methods, events, and classes
   * Listeners — Objects that respond to a specific event by executing a callback function when the event notice is broadcast
   * Objects — Instances of classes, which contain actual data values stored in the objects' properties
   * Subclasses — Classes that are derived from other classes and that inherit the methods, properties, and events from those classes (subclasses facilitate the reuse of code defined in the superclass from which they are derived).
   * Superclasses — Classes that are used as a basis for the creation of more specifically defined classes (that is, subclasses).
   * Packages — Folders that define a scope for class and function naming
8. procedural programming

## misc00

1. overload
   * `eq(), ==`
   * `plus(), +`
2. functions to test objects
   * `isa(zc1, 'timeseries')`: specific class
   * `isobject()`: `False` for `numeric, logical, char, cell, struct, and function handle``
   * `eq(), ==`: the same handle
   * `isequal()`: different handles have equal property values; the same class, size, property values
3. query class components
   * `class()`
   * `enumeration()`
   * `events()`
   * `methods()`
   * `methodsview()`
   * `properties()`

## handle class

1. copy of handles
2. handle objects modified in function
3. **NO** use `ishandle()`
4. `isa(zc1, 'handle')`
5. `isvalid(zc1)`
6. `deleted(zc1)`, `clear zc1` (not necessary)

```MATLAB
load gong.mat Fs y
% load('handel.mat','Fs','y')
% zc1 = timeseries(rand(100,1),.01:.01:1,'Name','Data1');
zc1 = audioplayer(y,Fs);

% class(zc1)
% play(zc1)
zc2 = zc1
disp(zc2.SampleRate) %8192
zc1.SampleRate = zc1.SampleRate*2;
disp(zc2.SampleRate) %16384
```

## Create a Simple Class

see [link](https://www.mathworks.com/help/releases/R2017b/matlab/matlab_oop/create-a-simple-class.html)

1. keyword
   * `classdef...end`: all class components
   * `properties...end`: property names, attributes, default value
   * `methods...end`: method signature, attributes, code
   * `events...end`: event name, attribute
   * `enumeration...end`:enumeration members, values
2. Constructor
3. vectorization, array of object
   * `[obj.value]+1`
   * `method(obj)`: `[obj.value]` in `method()` function
   * `z1.method()`
4. overload functions
   * `plus(obj)`
   * `multiply(obj1,obj2)`
5. `classdef` block
   * `<`: supper class
   * `(Sealed)`: cannot be used to derive subclass
6. `properties` block
   * `(Access = private)`
   * default value
7. `methods` block
   * `(Access = private)`
8. `events` block
   * `(ListenAccess = protected)`

```matlab
classdef (Sealed) Class01 < handle
    properties (Access = private)
        Prop1 = date % function call
    end
    methods (Access = private)

    end
    events (ListenAccess = protected)
        stateChanged
    end
end
```
