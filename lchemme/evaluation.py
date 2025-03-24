
from schemist.tables import converter
from transformers import pipeline


def get_model_smiles(df, column, pipe, batch_size=128):

    ds = Dataset.from_dict(df[[column]].to_dict("list"))
    results = pipe(KeyDataset(ds, column), batch_size=batch_size)

    return [item['translation_text'] for d in results for item in d]


def check_canonicalization_performance(df, column, batch_size=128, sample_size=10_000):
    
    canonicalizer = pipeline("translation", 
                             model=LLM_PATH, #llm_model, 
                             # tokenizer=tokenizer,
                             max_length=500,
                             device='cuda')

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
                           output_representation=['permuted_smiles', 'inchikey'])
    errors, df = converter(df, 'model_smiles', 
                           output_representation='inchikey',
                           prefix='model_')
    print_err(errors)
    
    df = df.assign(model_correct_canonical_smiles=lambda x: x[canon_col] == x['model_smiles'],
                   model_correct_molecule=lambda x: x['model_inchikey'] == x['inchikey'])
    mean_correct_mol = df["model_correct_molecule"].mean()
    mean_correct_smiles = df["model_correct_canonical_smiles"].mean()

    print_err("Proportion correct canonical SMILES:", mean_correct_smiles)
    print_err("Proportion correct molecule:", mean_correct_mol)

    return df