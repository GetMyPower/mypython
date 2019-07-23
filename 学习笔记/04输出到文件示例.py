with open("test.csv","w") as f:
    f.writelines("车的id,SOC(%),开始充电时刻Tws(h),充电时长Twc(h),充电结束时刻Tce(h),Pc(kW)\n")
    for tmp in Car_Park:
        f.writelines(str(tmp.id)+","+str(tmp.E0)+","+str(tmp.Tws)+","+str(tmp.Twc)+","+str(tmp.Tce)+","+str(tmp.Pc)+"\n")