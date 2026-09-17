"""عرض/تنزيل Dataset من صفحة المصدر الرسمية أو رابط مباشر."""
import argparse,hashlib,json,re,sys,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RAW=ROOT/'data'/'raw';SOURCE='https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets'
def page():
 r=urllib.request.Request(SOURCE,headers={'User-Agent':'PowerGridCyberAI/1.0'});return urllib.request.urlopen(r,timeout=60).read().decode('utf-8','replace')
def links():
 h=page();return re.findall(r'href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',h,re.I|re.S)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--list',action='store_true');ap.add_argument('--url');ap.add_argument('--name');ap.add_argument('--extract',action='store_true');a=ap.parse_args();
 if a.list:
  for i,(u,l) in enumerate(links(),1):
   if any(x in (u+l).lower() for x in ['download','.csv','.zip','.arff','drive.google']):print(i,re.sub('<[^>]+>',' ',l).strip(),u)
  return
 if not a.url:ap.error('استخدم --list ثم --url DIRECT_URL')
 RAW.mkdir(parents=True,exist_ok=True);name=a.name or Path(a.url.split('?')[0]).name or 'dataset_download';dest=RAW/name;req=urllib.request.Request(a.url,headers={'User-Agent':'PowerGridCyberAI/1.0'});data=urllib.request.urlopen(req,timeout=120).read();
 if b'<html' in data[:500].lower():raise ValueError('الرابط يعيد HTML وليس ملفاً مباشراً')
 dest.write_bytes(data);digest=hashlib.sha256(data).hexdigest();print('Saved',dest,'SHA256',digest)
 if a.extract or dest.suffix.lower()=='.zip':
  with zipfile.ZipFile(dest) as z:z.extractall(RAW)
 (RAW/'download_manifest.json').write_text(json.dumps({'source_page':SOURCE,'url':a.url,'file':str(dest.relative_to(ROOT)),'sha256':digest},indent=2),encoding='utf-8')
if __name__=='__main__':
 try:main()
 except Exception as e:print('ERROR:',e,file=sys.stderr);raise SystemExit(1)
