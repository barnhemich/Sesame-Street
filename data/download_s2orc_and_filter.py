"""


Example of how one would download & process a single batch of S2ORC to filter to specific field of study.
Can be useful for those who can't store the full dataset onto disk easily.
Please adapt this to your own field of study.


Creates directory structure:

|-- metadata/
    |-- raw/
        |-- metadata_0.jsonl.gz      << input; deleted after processed
    |-- medicine/
        |-- metadata_0.jsonl         << output
|-- pdf_parses/
    |-- raw/
        |-- pdf_parses_0.jsonl.gz    << input; deleted after processed
    |-- medicine/
        |-- pdf_parses_0.jsonl       << output

"""


import os
import subprocess
import gzip
import io
import json
from tqdm import tqdm
import re


# process single batch
def process_batch(batch: dict):
    # this downloads both the metadata & full text files for a particular shard
    print(batch['input_metadata_path'])
    print(batch['input_metadata_url'])
    cmd = ["wget", "-O", batch['input_metadata_path'], batch['input_metadata_url']]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    print(batch['input_pdf_parses_path'])
    print(batch['input_pdf_parses_url'])
    cmd = ["wget", "-O", batch['input_pdf_parses_path'], batch['input_pdf_parses_url']]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    # first, let's filter metadata JSONL to only papers with a particular field of study.
    # we also want to remember which paper IDs to keep, so that we can get their full text later.
    paper_ids_to_keep = set()
    with gzip.open(batch['input_metadata_path'], 'rb') as gz, open(batch['output_metadata_path'], 'wb') as f_out:
        f = io.BufferedReader(gz)
        for line in tqdm(f.readlines()):
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            mag_field_of_study = metadata_dict['mag_field_of_study']
            if mag_field_of_study and 'Computer Science' in mag_field_of_study:     # TODO: <<< change this to your filter
                paper_ids_to_keep.add(paper_id)
                f_out.write(line)

    # now, we get those papers' full text
    with gzip.open(batch['input_pdf_parses_path'], 'rb') as gz, open(batch['output_pdf_parses_path'], 'wb') as f_out:
        f = io.BufferedReader(gz)
        for line in tqdm(f.readlines()):
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            if paper_id in paper_ids_to_keep:
                f_out.write(line)

    # now delete the raw files to clear up space for other shards
    os.remove(batch['input_metadata_path'])
    os.remove(batch['input_pdf_parses_path'])


if __name__ == '__main__':

    METADATA_INPUT_DIR = 'F:/cs2/metadata/raw/'
    METADATA_OUTPUT_DIR = 'F:/cs2/metadata/cs/'
    PDF_PARSES_INPUT_DIR = 'F:/cs2/pdf_parses/raw/'
    PDF_PARSES_OUTPUT_DIR = 'F:/cs2/pdf_parses/cs/'

    os.makedirs(METADATA_INPUT_DIR, exist_ok=True)
    os.makedirs(METADATA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PDF_PARSES_INPUT_DIR, exist_ok=True)
    os.makedirs(PDF_PARSES_OUTPUT_DIR, exist_ok=True)

    # TODO: make sure to put the links we sent to you here
    # there are 100 shards with IDs 0 to 99. make sure these are paired correctly.
    # download_linkss = [
    #     {"metadata": "https://...", "pdf_parses": "https://..."},  # for shard 0
    #     {"metadata": "https://...", "pdf_parses": "https://..."},  # for shard 1
    #     {"metadata": "https://...", "pdf_parses": "https://..."},  # for shard 2
    # ]
    # download_linkss = [
    #     {"metadata": "https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_0.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=uPfuXHEVYI39hwwibohK1CcK5Q0%3D&Expires=1654817320", "pdf_parses": "https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_0.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=tM6gFLX%2BKuMv0R2jkA%2FWB8Ht0qM%3D&Expires=1654817323"},  # for shard 0
    # ]
    # download_linkss = []
    # for i in range(100):
    #     download_linkss.append({"metadata": f"https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/metadata/metadata_{i}.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=uPfuXHEVYI39hwwibohK1CcK5Q0%3D&Expires=1654817320", "pdf_parses": f"https://ai2-s2-s2orc.s3.amazonaws.com/20200705v1/full/pdf_parses/pdf_parses_{i}.jsonl.gz?AWSAccessKeyId=AKIA5BJLZJPW4OD5EQ2P&Signature=tM6gFLX%2BKuMv0R2jkA%2FWB8Ht0qM%3D&Expires=1654817323"})

    md = []
    pdf = []
    with open("dl_s2orc_20200705v1_full_urls_expires_20220609.txt") as f:
        for line in f:
            if "metadata/metadata_" in line:
                link = re.search("https.+$", line)
                md.append(link[0][:-1])
            elif "pdf_parses/pdf_parses_" in line:
                link = re.search("https.+$", line)
                pdf.append(link[0][:-1])
    download_linkss = []
    for i, j in zip(md, pdf):
        download_linkss.append({"metadata": i, "pdf_parses": j})

    # turn these into batches of work
    # TODO: feel free to come up with your own naming convention for 'input_{metadata|pdf_parses}_path'
    batches = [{
        'input_metadata_url': download_links['metadata'],
        'input_metadata_path': os.path.join(METADATA_INPUT_DIR,
                                            os.path.basename(download_links['metadata'].split('?')[0])),
        'output_metadata_path': os.path.join(METADATA_OUTPUT_DIR,
                                             os.path.basename(download_links['metadata'].split('?')[0])),
        'input_pdf_parses_url': download_links['pdf_parses'],
        'input_pdf_parses_path': os.path.join(PDF_PARSES_INPUT_DIR,
                                              os.path.basename(download_links['pdf_parses'].split('?')[0])),
        'output_pdf_parses_path': os.path.join(PDF_PARSES_OUTPUT_DIR,
                                               os.path.basename(download_links['pdf_parses'].split('?')[0])),
    } for download_links in download_linkss]

    for batch in batches:
        process_batch(batch=batch)
