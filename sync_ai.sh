#php artisan serve --host=0.0.0.0 --port=8000 >/dev/null &

rsync -avz --delete --progress -e "ssh -i ~/.ssh/Linux-CodeCambat.pem" --exclude ~/git/datapai-streamlit/.claude/ --exclude ~/git/datapai-streamlit/.git/ --exclude ~/git/datapai-streamlit/agents/__pycache__/ --exclude ~/git/datapai-streamlit/.streamlit ~/git/datapai-streamlit/ ec2-user@platform.datap.ai:/home/ec2-user/git/vanna-streamlit/ 
#rsync -avL --delete --progress -e "ssh -i ~/.ssh/Linux-CodeCambat.pem"  --exclude "storage" ~/git/kalepa/*  ec2-user@test.datap.ai:/var/www/kalepa/

#npm run watch-poll 
