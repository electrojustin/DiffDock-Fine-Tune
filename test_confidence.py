import os
import shutil
import glob

if os.path.isdir('confidence_test_data'):
    shutil.rmtree('confidence_test_data')

os.system('python inference.py --config default_inference_args.yaml --protein_path examples/1a46_protein_processed.pdb --ligand_description examples/1a46_ligand.sdf --out_dir confidence_test_data/')

failures = []

for file in glob.glob('confidence_test_data/complex_0/rank*.sdf'):
    if file == 'confidence_test_data/complex_0/rank1.sdf':
        continue

    confidence = file[file.find('_confidence') + len('_confidence'):]
    confidence = confidence[:confidence.find('.sdf')]
    confidence = float(confidence)

    os.system('python infer_confidence.py --config default_confidence_args.yaml --protein_path examples/1a46_protein_processed.pdb --ligand_description examples/1a46_ligand.sdf --pose_path ' + file)
    with open('scores.tsv') as score_file:
        test_confidence = float(score_file.readlines()[0].split('\t')[1])
    if abs(confidence - test_confidence) > 0.1:
        failures.append(file, confidence, test_confidence)

if not failures:
    print('Test passed!')
else:
    print('Test failed!')
    for failure in failures:
        print(failure[0] + ':')
        print('expected score: ' + str(failure[1]))
        print('actual score: ' + str(failure[2]))
