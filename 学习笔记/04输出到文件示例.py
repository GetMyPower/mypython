with open("test.csv","w") as f:
    f.writelines("����id,SOC(%),��ʼ���ʱ��Tws(h),���ʱ��Twc(h),������ʱ��Tce(h),Pc(kW)\n")
    for tmp in Car_Park:
        f.writelines(str(tmp.id)+","+str(tmp.E0)+","+str(tmp.Tws)+","+str(tmp.Twc)+","+str(tmp.Tce)+","+str(tmp.Pc)+"\n")