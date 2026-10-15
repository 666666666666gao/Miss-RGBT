import csv,datetime,json,os,shutil,subprocess,time
from pathlib import Path
assert shutil.which('nvidia-smi')
def cpu():
 return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
previous=None
samples=[]
for index in range(6):
 current=cpu()
 record={'sample':index,'observed_utc':datetime.datetime.utcnow().isoformat()+'Z','load_average':list(os.getloadavg())}
 if previous is not None:
  d=[a-b for a,b in zip(current,previous)]
  total=sum(d)
  record['cpu_busy_percent']=100*(total-d[3]-d[4])/total
  record['cpu_iowait_percent']=100*d[4]/total
 previous=current
 memory={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:','MemAvailable:'))}
 record['host_memory_bytes']=memory
 query=subprocess.check_output(['nvidia-smi','-i','2,3','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory','--format=csv,noheader,nounits'],universal_newlines=True)
 record['gpus']=[]
 for row in csv.reader(query.splitlines()):
  gpu,name,total,used,free,util,memory_util=[v.strip() for v in row]
  assert int(gpu) in [2,3]
  record['gpus'].append({'physical_gpu':int(gpu),'name':name,'total_mib':int(total),'used_mib':int(used),'free_mib':int(free),'gpu_util_percent':int(util),'memory_util_percent':int(memory_util)})
 assert sorted(v['physical_gpu'] for v in record['gpus'])==[2,3]
 samples.append(record)
 if index<5: time.sleep(12)
print(json.dumps({'logical_cpu_count':os.cpu_count(),'sample_count':len(samples),'nominal_duration_seconds':60,'samples':samples,
 'scope':'Requested GPU2/3 memory/utilization and aggregate CPU/RAM only. No temperature, power, other GPU, task progress, or neural execution.'}))
