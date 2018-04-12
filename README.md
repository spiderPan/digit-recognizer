To pull training data
```
kaggle competitions download -c digit-recognizer -p ./data
```
To start training
```
python ./run.py
```
After training, to submit data
```
kaggle competitions submit -c digit-recognizer -f ./data/submission.csv -m "Message"
```
