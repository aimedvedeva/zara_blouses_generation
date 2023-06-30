import sys
import subprocess
import requests
import urllib.parse
import json

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-r",
    "/opt/ml/processing/input/requirements.txt",
])

from scrapy.http import TextResponse
from scrapy import Spider

API_KEY = '92744398-93ae-4cbc-ab11-f69bc621ba2e'

def get_scrapeops_url(url):
    payload = {'api_key': API_KEY, 'url': url}
    proxy_url = 'https://proxy.scrapeops.io/v1/?' + urllib.parse.urlencode(payload)
    return proxy_url

page = 1

while True:
  url = 'https://www.zara.com/es/es/mujer-camisas-l1217.html?v1=2184370&page='+str(page)
  r = requests.get(get_scrapeops_url(url))
  resp = TextResponse(body=r.content, url=url)
  data = resp.css("script[type='application/ld+json']::text").get()
  elements = json.loads(data)['itemListElement']

  for idx, item in enumerate(elements):
        
    if page == 6 and idx == 8:
      continue
        
    image_name = item['name']
    image_url = item['image']
    try:
      img_data = requests.get(image_url).content
      img_name = '/opt/ml/processing/output/'+'page_'+str(page)+'_idx_'+str(idx)+'_name_'+image_name+'.jpg'
      with open(img_name, 'wb') as handler:
        handler.write(img_data)
    except:
      print(image_url)
  if len(elements) > 0:
    page += 1
  else:
    break
