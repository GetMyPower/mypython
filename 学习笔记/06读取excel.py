import xlrd

data = xlrd.open_workbook('路线.xlsx') # 打开xls文件
table = data.sheets()[0] # 打开第二张表
nrows = table.nrows      # 获取表的行数

with open("出行.txt","w") as f:
    for col in range(2,4+1):   #col对应考点
        f.writelines("\n考点"+str(col-1)+"："+table.row_values(1)[col]+"\n")
        for frm in range(0,3+1):   #frm对应出发点0~3
            f.writelines("出发点" +str(  frm +1)+"："+table.row_values(2+  frm *3)[0]+"\n")
            for k in range(3):    #k对应3种出行方式
                f.writelines(table.row_values(2+frm *3+k)[1]+"：")    #出行方式名称
                f.writelines(table.row_values(2 +frm*3 + k)[col] + "\n")    #具体方案
            f.writelines("\n")
f.close()

print(table.row_values(3)[3])