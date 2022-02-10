%% run(test_test())
function ret = test_test()
    ret = functiontests(localfunctions);
end

function test00(testCase)
    assert(0 < 1);
end
