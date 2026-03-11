run:
	sudo modprobe -r v4l2loopback 2>/dev/null || true
	sudo modprobe v4l2loopback devices=1 video_nr=2 card_label="Ubermensh" exclusive_caps=1

	python3 ubermensh.py \
	  --device /dev/video2 \
	  --freq 100.1M \
	  --sample-rate 2.048M \
	  --gain auto \
	  --width 640 \
	  --height 480 \
	  --fps 30
