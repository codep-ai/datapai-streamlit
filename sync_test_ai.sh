#php artisan serve --host=0.0.0.0 --port=8000 >/dev/null &

rsync -avz --delete --progress -e "ssh -i ~/.ssh/Linux-CodeCambat.pem"  ~/git/datapai-streamlit/ ec2-user@test.datap.ai:/home/ec2-user/git/vanna-streamlit/ 
#rsync -avL --delete --progress -e "ssh -i ~/.ssh/Linux-CodeCambat.pem"  --exclude "storage" ~/git/kalepa/*  ec2-user@test.datap.ai:/var/www/kalepa/

#npm run watch-poll 
