kill -9 `ps -ef|grep 23333|grep -v grep|awk '{print $2}'`
nohup python manage.py runserver --settings=idcard_ocr.settings_staging 127.0.0.1:3335 &