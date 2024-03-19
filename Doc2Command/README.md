# Doc2Command

This repository contains code for Doc2Command, a multi task model to generate machine parsable commands and region of interest boxes based on an open vocabulary user request and a document image.

<img width="100%" alt="image" src="https://github.com/MananSuri27/doc2command/assets/84636031/812bc8c5-6624-4d04-be14-357fca6da483">

## Setup

Download the data from [here](https://github.com/adobe-research/DocEdit-Dataset/releases/tag/v1.0) / or use the script in `\data`.

Install dependencies:
```bash
pip install -r requirements.txt
```

To run train the model, you can use `main.py`. You can configure arguments from the argument parser in the script.

```bash
python3 main.py 
```

