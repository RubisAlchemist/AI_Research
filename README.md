# AI Research

time series dataset -> score of mental illness

## files
- model.py : time series model, train code
  - SiMBASimple, SiMBA model
- simba.py : SiMBA code from https://github.com/badripatro/simba
- handwriting.py : model, train code
  - goal : handwriting letter classification from time series data
- tac.py : model, train code
  - goal : TAC score regression from time series accelerometer data
- mindlamp.py : model, train code
  - goal : regression mental illness score from time series data
## how to run

prepare dataset
### dataset
- ./dataset/handwriting (preapred)  
- ./dataset/tac
- ./dataset/lamp   

### script
```bash
# env setting
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# run code
python run.py handwriting --batch 32 --epochs 12 --lr 0.0003 --simple
python run.py tac --seq_len 1000 --batch 32 --epochs 12 --lr 0.003 --simple
python run.py mindlamp_r --target 'PHQ-9' --batch 8 --seq_len 6000 --epochs 12 --lr 0.003 --simple
python run.py mindlamp_c --target 'is_PHQ-9' --batch 8 --seq_len 6000 --epochs 12 --lr 0.003 --simple
```

## result

### handwriting classfication
- batch=32, epochs=12, lr=3e-4, model=SiMBA
  - best val accuracy : 91.33%, val loss : 0.4158

### tac regression
- batch=32, epochs=12, lr=3e-3, seq_len=1000, model=SiMBA

### mindlamp regression 
