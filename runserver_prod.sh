kill -9 `ps -ef|grep 23333|grep -v grep|awk '{print $2}'`
nohup python manage.py runserver --settings=idcard_ocr.settings_prod 127.0.0.1:23335 &