import sys
import os
import time
import io
import traceback
import nbformat
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8')

NOTEBOOK_PATH = Path('notebook/RE-FUSED.ipynb')
LOG_PATH = Path('run_notebook_progress.log')

def log(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    with open(LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(formatted + '\n')

def run_all_cells():
    if LOG_PATH.exists():
        LOG_PATH.unlink()
        
    log("======================================================================")
    log(f"Starting Notebook Execution: {NOTEBOOK_PATH}")
    log("======================================================================")
    
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    total_cells = len(nb.cells)
    code_cell_count = sum(1 for c in nb.cells if c.cell_type == 'code')
    log(f"Notebook loaded: {total_cells} total cells ({code_cell_count} code cells)")

    exec_globals = {'__name__': '__main__'}
    exec_count = 1
    t_start = time.time()
    
    results_summary = []

    for idx, cell in enumerate(nb.cells):
        cell_id = cell.get('id', f'cell_{idx}')
        if cell.cell_type == 'markdown':
            lines = cell.source.strip().split('\n')
            title = [l.strip('# ').strip() for l in lines if l.startswith('#')]
            t_str = title[0] if title else lines[0][:50] if lines else "Markdown"
            log(f"\n--- [Cell {idx:02d}/{total_cells-1}] Markdown: {t_str} ---")
            continue

        # Code cell execution
        first_line = cell.source.strip().split('\n')[0] if cell.source.strip() else 'Empty cell'
        log(f"\n>>> [Cell {idx:02d}/{total_cells-1} | Code Exec #{exec_count}] Running: {first_line[:75]}...")
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        
        sys.stdout = captured_out
        sys.stderr = captured_err
        
        cell_t0 = time.time()
        exec_error = None
        
        try:
            exec(cell.source, exec_globals)
        except Exception as e:
            exec_error = traceback.format_exc()
        
        cell_dt = time.time() - cell_t0
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        stdout_str = captured_out.getvalue()
        stderr_str = captured_err.getvalue()
        
        cell_outputs = []
        if stdout_str:
            cell_outputs.append(nbformat.v4.new_output(
                output_type='stream',
                name='stdout',
                text=stdout_str
            ))
        if stderr_str:
            cell_outputs.append(nbformat.v4.new_output(
                output_type='stream',
                name='stderr',
                text=stderr_str
            ))
        if exec_error:
            cell_outputs.append(nbformat.v4.new_output(
                output_type='error',
                ename=type(e).__name__,
                evalue=str(e),
                traceback=exec_error.split('\n')
            ))
            
        cell.outputs = cell_outputs
        cell.execution_count = exec_count
        
        # Save notebook state after each code cell
        nbformat.write(nb, NOTEBOOK_PATH)
        
        status_symbol = "❌ FAILED" if exec_error else "✅ PASSED"
        log(f"[{status_symbol}] Cell {idx:02d} finished in {cell_dt:.2f}s (Exec #{exec_count})")
        
        if stdout_str.strip():
            first_lines = [l for l in stdout_str.strip().split('\n') if l.strip()]
            for l in first_lines[:5]:
                log(f"   | {l}")
            if len(first_lines) > 5:
                log(f"   | ... ({len(first_lines)-5} more output lines)")
                
        results_summary.append({
            'index': idx,
            'exec_count': exec_count,
            'title': first_line[:60],
            'duration': cell_dt,
            'passed': exec_error is None,
            'error': exec_error
        })
        
        exec_count += 1
        
        if exec_error:
            log(f"\n❌ Execution stopped due to error in Cell {idx:02d}:\n{exec_error}")
            break

    total_time = time.time() - t_start
    passed_count = sum(1 for r in results_summary if r['passed'])
    failed_count = sum(1 for r in results_summary if not r['passed'])

    log("\n======================================================================")
    log(f"EXECUTION SUMMARY: {passed_count}/{code_cell_count} code cells passed in {total_time:.2f}s ({total_time/60:.2f} min)")
    log("======================================================================")
    
    return results_summary

if __name__ == '__main__':
    run_all_cells()
