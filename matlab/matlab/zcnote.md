# misc notes

## 有趣的代码

1. mail2me, "MATLAB 对我说——MATLAB 中的程序运行状态提醒 myisland，USST", [link](http://www.matlabsky.com/thread-25833-1-1.html)
2. flappy bird
3. grabit
4. `export_fig`

## 有趣的代码-grabit

鼠标的操作

1. click (为了double click的实现，禁止使用normal + single click的响应)
2. ctrl + click / left+right click
3. shift + click
4. double click
5. 不考虑鼠标方便的但触摸板不方便的操作

鼠标要实现的功能

| operate | mouse | keyboard |
| :-: | :-: | :-: |
| 加载图像 | (NPS)double click | l |
| 平移 | (PM)double click + motion | ctrl/normal/shift + left/right/up/down |
| zoom out | (ZM)double click + motion | + |
| zoom in | | - |
| reset view | (ZM)double click | _ |
| 选点 | shift + click | |
| transfor mode | right click / ctrl + click | t |
| help | | h |
| delete point | | delete |
| save point to workspace | | enter |

mode

| state | operate | state |
| :---: | :----: | :------:|
| pan mode (PM) | ctrl + click / double click | zoom mode |
| zoom mode (ZM) | ctrl + click / double click | pan mode |

状态流

1. no picture state(NPS)
2. 还没想清楚

窗口

1. `hFig`：接受 mouse/keyboard 指令的唯一对象
2. `hAxes`
3. `hLine`
4. `hUictrol_modeView`
5. `hUictrol_help`: 一段时间后消失，透明度处理
