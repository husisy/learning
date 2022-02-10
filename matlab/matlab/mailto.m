function mailto(subject,content,targetMail)
if nargin < 3
    targetMail = '****@mail.ustc.edu.cn';
end
MailAddress = 'xxx@qq.cn'; % replace xxx with true mail address
password = 'xxx'; %replace xxx with true password
setpref('Internet','E_mail',MailAddress);
setpref('Internet','SMTP_Server','smtp.189.cn');
setpref('Internet','SMTP_Username',MailAddress);
setpref('Internet','SMTP_Password',password);
props = java.lang.System.getProperties;
props.setProperty('mail.smtp.auth','true');
sendmail(targetMail,subject,content);
end
