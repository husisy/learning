# nni

1. link
   * [github](https://github.com/microsoft/nni)
   * [documentation](https://nni.readthedocs.io/en/latest/index.html)
2. installation `pip install nni`
3. `config.yaml`
   * `useActiveGpu: false`
4. tuner, assessor
5. standalone mode for debugging

```bash
nnictl create --config config.yml
nnictl experiment show #show the information of experiments
nnictl trial ls #list all of trial jobs
nnictl top #monitor the status of running experiments
nnictl log stderr #show stderr log content
nnictl log stdout #show stdout log content
nnictl stop #stop an experiment
nnictl trial kill #kill a trial job by id
nnictl --help #get help information about nnictl
```
