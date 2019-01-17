# Named Entity Recognition using Conditional Random Fields
##### Created by Peter Simkin (DolphinDance) & Tim Avni (tavni96)

```python
from DIYNER.model import CRFNER

# Gazateer = {'entity': 'category'}
# Documents = ['doc1','doc2','doc3']

# Training Model
ner_crf = CRFNER(gazetteer)
ner_crf.train(documents)

# Predictions
ner_crf.predict('Google are opening new stores in Vancouver')
[['company', '0', '0', '0', '0', '0', 'city']]
```

### Visualisation
```python
from DIYNER.viz import radialTree
```

### Example Output
![Radial](https://i.imgur.com/oC2Jitu.png)