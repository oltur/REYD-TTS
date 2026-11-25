SERVER_USER := ot
SERVER_HOST := 192.168.68.92
DEPLOY_DIR  := /home/ot/reyd-tts-deploy

.PHONY: deploy deploy-cloudflared

deploy:
	rsync -avz --exclude='.git' --exclude='.gitignore' --exclude='__pycache__' \
		--exclude='.DS_Store' --exclude='.cloudflared_token' \
		--exclude='FastSpeech2' \
		./ $(SERVER_USER)@$(SERVER_HOST):$(DEPLOY_DIR)/
	ssh -t $(SERVER_USER)@$(SERVER_HOST) "sudo bash $(DEPLOY_DIR)/server_setup.sh"

deploy-cloudflared:
	scp cloudflared_setup.sh $(SERVER_USER)@$(SERVER_HOST):/home/ot/cloudflared_setup.sh
	TOKEN=$$(cat .cloudflared_token); \
	ssh -t $(SERVER_USER)@$(SERVER_HOST) "sudo bash /home/ot/cloudflared_setup.sh '$$TOKEN'"
