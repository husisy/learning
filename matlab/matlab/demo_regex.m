str = 'She sells sea shells by the seashore.';
expression = '[Ss]h.';
[match,noMatch] = regexp(str,expression,'match','split');
combinedStr = strjoin(noMatch,match);


str = '<title>My Title</title><p>Here is some text.</p>';
expression = '<(\w+).*>.*</\1>';
[tokens,matches] = regexp(str,expression,'tokens','match');


str = '01/11/2000  20-02-2020  03/30/2000  16-04-2020';
expression = ['(?<month>\d+)/(?<day>\d+)/(?<year>\d+)|'...
              '(?<day>\d+)-(?<month>\d+)-(?<year>\d+)'];
tokenNames = regexp(str,expression,'names');


expression = {'(?-i)\w*case';...
              '(?i)\w*case'};
matchStr = regexp(str,expression,'match');
celldisp(matchStr)
