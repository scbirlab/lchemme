
from carabiner import print_err
from datasets import Dataset
from schemist.tables import converter
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

import torch

def get_model_smiles(df, column, pipe, batch_size=128):

    ds = Dataset.from_dict(df[[column]].to_dict("list"))
    results = pipe(KeyDataset(ds, column), batch_size=batch_size)

    return [item['translation_text'] for d in results for item in d]


def check_canonicalization_performance(
    df, 
    column,
    model_id: str,
    batch_size: int = 128, 
    sample_size: int = 10_000
):
    # TODO: Make this actually functional
    canonicalizer = pipeline("translation", 
                             model=model_id, #llm_model, 
                             # tokenizer=tokenizer,
                             max_length=500,
                             device='cuda' if torch.cuda.is_available() else 'cpu')

    if df.shape[0] > sample_size:
        df = df.sample(min(df.shape[0], sample_size))

    df = df.assign(
        model_smiles=lambda x: get_model_smiles(
            x, 
            column, 
            pipe=canonicalizer, 
            batch_size=batch_size,
        )
    )
    
    errors, df = converter(df, column, 
                           output_representation=['smiles', 'permuted_smiles', 'inchikey'])
    errors, df = converter(df, 'model_smiles', 
                           output_representation='inchikey',
                           prefix='model_')
    print_err(errors)
    
    df = df.assign(model_correct_canonical_smiles=lambda x: x['smiles'] == x['model_smiles'],
                   model_correct_molecule=lambda x: x['model_inchikey'] == x['inchikey'])
    mean_correct_mol = df["model_correct_molecule"].mean()
    mean_correct_smiles = df["model_correct_canonical_smiles"].mean()

    print_err("Proportion correct canonical SMILES:", mean_correct_smiles)
    print_err("Proportion correct molecule:", mean_correct_mol)

    return df