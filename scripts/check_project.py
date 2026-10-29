"""Validate starter integrity; this is NOT a completed-lab/content grader."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute-notebooks',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'data/raw/manifest.json').read_text(encoding='utf-8'))
    raw=ROOT/'data/raw'/manifest['filename']
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==manifest['sha256'],'Raw hash mismatch'
    pdf=ROOT/'references/excercise_ML_models.pdf'
    assert hashlib.sha256(pdf.read_bytes()).hexdigest()==manifest['source_pdf_sha256'],'PDF hash mismatch'
    members=json.loads((ROOT/'docs/team.json').read_text(encoding='utf-8'))
    assert len(members)==6 and len({m['id'] for m in members})==6
    for m in members: assert (ROOT/m['notebook']).is_file()
    import re
    cfg=json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    assert cfg['planning_granularity']=='day' and cfg['collaboration_mode']=='asynchronous'
    assert cfg['team_work_start_date']=='2026-10-30'
    assert cfg['data_preparation_owner']=='assistant' and cfg['data_preparation_status']=='ready'
    assert 'internal_code_freeze' not in cfg and 'internal_ready_to_submit' not in cfg
    for key in ('feature_freeze_date','model_config_freeze_date','internal_code_freeze_date','internal_ready_to_submit_date'):
        assert re.fullmatch(r'2026-\d{2}-\d{2}',cfg[key]),key
    plan=(ROOT/'docs/TEAM_PLAN.md').read_text(encoding='utf-8')
    assert not re.search(r'\b\d{1,2}:\d{2}\b',plan),'Fixed clock slots remain'
    for member in members:
        assert member['schedule_unit']=='day' and 'effort_hours' not in member
        assert member['work_start_date']=='2026-10-30' and member['first_handoff_date']>='2026-10-30'
        assert re.fullmatch(r'2026-\d{2}-\d{2}',member['section_complete_date'])
        note=json.loads((ROOT/member['notebook']).read_text(encoding='utf-8'))
        assert note['metadata']['lab3']['schedule_unit']=='day'
        handoff=(ROOT/'docs/handoffs'/f"{member['id']}.md").read_text(encoding='utf-8')
        assert 'ngày nào' in handoff and 'giờ nào' not in handoff
    assert cfg['test_start']=='2015-10-27 05:00:00' and cfg['warmup_hours']==168

    for f in list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py')):
        ast.parse(f.read_text(encoding='utf-8'),filename=str(f))
    import nbformat
    notebooks=sorted((ROOT/'notebooks').glob('*.ipynb'))
    assert len(notebooks)==6
    executed=0
    for file in notebooks:
        nb=nbformat.read(file,as_version=4);nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':
                ast.parse(cell.source,filename=file.name)
                assert not any(o.output_type=='error' for o in cell.outputs),'Stored traceback'
        if args.execute_notebooks:
            from nbclient import NotebookClient
            from jupyter_client.kernelspec import KernelSpecManager
            with tempfile.TemporaryDirectory() as tmp:
                spec=Path(tmp)/'lab3-check';spec.mkdir()
                (spec/'kernel.json').write_text(json.dumps({
                    'argv':[sys.executable,'-m','ipykernel_launcher','--log-level=ERROR','-f','{connection_file}'],
                    'display_name':'Lab3 scaffold check','language':'python'}),encoding='utf-8')
                ksm=KernelSpecManager(kernel_dirs=[tmp])
                from jupyter_client import KernelManager
                km=KernelManager(kernel_name='lab3-check',kernel_spec_manager=ksm)
                NotebookClient(nb,timeout=180,km=km,
                    resources={'metadata':{'path':str(ROOT/'notebooks')}}
                    ).execute(cleanup_kc=True)
                executed+=1  # execute in memory: keep distributed starter outputs clean
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    result=unittest.TestResult();suite.run(result)
    if not result.wasSuccessful():
        for test,trace in result.errors+result.failures:print(str(test)+'\n'+trace,file=sys.stderr)
        return 1
    print('SCHEDULE_OK mode=day members=6 dates=5')
    print(f'MODIFIED_OK tests={result.testsRun} notebooks={len(notebooks)} executed={executed} mode=scaffold')
    return 0
if __name__=='__main__':
    raise SystemExit(main())
