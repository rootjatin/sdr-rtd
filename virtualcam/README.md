for rtl std : <br>
python3 sdr_canvas.py --device /dev/video2 --freq 100e6 --rate 2.4e6 --gain -1 \
  --fft 2048 --avg 8 --overlap 0.5 --auto-range --preview

Peak hold (nice for sweeping a slewable dish): <br>
python3 sdr_canvas.py --peak-hold --peak-decay 0.99 --auto-range --preview


HackRF One general use : <br> 
python3 sdr_virtual_cam.py --backend soapy --soapy-args "driver=hackrf" \
  --device /dev/video2 --freq 100e6 --rate 10e6 --gain 20 \
  --fft 4096 --avg 10 --overlap 0.5 \
  --auto-range --preview
<br>
peak hold <br>
python3 sdr_virtual_cam.py --backend soapy --soapy-args "driver=hackrf" \
  --device /dev/video2 --freq 100e6 --rate 10e6 --gain 20 \
  --fft 4096 --avg 10 --overlap 0.5 \
  --auto-range --peak-hold --peak-decay 0.99 --preview
