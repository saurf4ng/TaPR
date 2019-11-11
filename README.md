
# Time-series Aware Precision and Recall

This script is developed on Python3.6.


## Input files (encoded by UTF-8)

This script supports two kinds of file format. (Assume that normal and anomaly labels are '1' and '-1', respectively.)

1. Label series: each line consists of [(label)('\n')] as belows:

```
1
-1
-1
1
1
```

2. Range: each line consists of [(first_index),(last_index),(attack_name)('\n')] as belows:
          For example, ./sample/Seq2Seq.csv follows this format.

```
1754,2800,1
3080,3690,2
5299,5400,3
```


## Running

python ./TaPR.py -i <prediction_file> -c <anomaly_file> {-a} <alpha> {-t} <theta> {-d} <delta> {-l} <label> {-p}

Here is a description of all command line options, inputs, and parameters:

```
-i: File with predictions
-c: File with anomalies (i.e., ground truth)
-t: Parameter theta for detection scoring 
	Set as float value from 0 to 1
    Default = 0.5
-d: Parameter delta for subsequent scoring
	Set as zero or more larger integer value
    Defualt = 0
-l: Normal and anomaly labels
	Set as two integers separate by ','
    Default = 1,-1
-p: Enable printing the list of detected anomalies and correct predictions
    No need input values 
-a: Parameter alpha indicating weight for the detection score
    Default = 0.5
```

If you need to see help menue, please type below operation:

```
python ./TaPR.py -h
```


## Examples

Below two examples produce indentical results.

```
python ./TaPR.py -i sample/ocsvm.csv -a sample/swat.csv -t 0.5 -d 180 -l 1,-1
```

```
python ./TaPR.py -i sample/ocsvm.csv -a sample/swat.csv -d 180
```

## Cite
The details of this work is going to be published on the 28th international conference on Information and Knowledge Management (CIKM) 2019 as belows.

```
@inproceedings{hwang2019time,
  title={Time-Series Aware Precision and Recall for Anomaly Detection: Considering Variety of Detection Result and Addressing Ambiguous Labeling},
  author={Hwang, Won-Seok and Yun, Jeong-Han and Kim, Jonguk and Kim, Hyoung Chun},
  booktitle={Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  pages={2241--2244},
  year={2019},
  organization={ACM}
}
```
