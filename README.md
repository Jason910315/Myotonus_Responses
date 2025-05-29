本研究旨在建構一套非侵入式、低成本且快速操作之肌肉狀態智能評估系統，以解決現行臨床肌力測量普遍存在的主觀性高、操作複雜與儀器昂貴等問題。透過收集肌肉施力過程中產生的聲波訊號，搭配機器學習模型，預測關鍵肌肉指標（如 Stiffness、Torque），作為肌力評估、復健追蹤與運動表現監控之依據。  

🔁 系統實驗與分析流程  
1. 實驗設計與數據收集 :
   使用特製肌力量測裝置（測量 Torque 與 Stiffness）
   每位受試者進行不同收縮姿勢測試，同步錄製聲音訊號

2. 聲學特徵擷取與資料處理 :
   使用 warbleR 對原始 .wav 音訊擷取 26 種特徵，後續進行標準化、清洗與特徵選擇

3. 模型訓練與驗證 :
   切分訓練集與測試集，建立多種回歸模型，並使用 RMSE 與 R² 進行比較與調參，最佳模型為 Gradient Boosting（RMSE = 0.91、R² = 0.91）

系統架構圖  
<img src="https://github.com/Jason910315/Myotonus_Responses/blob/main/system_flow.jpg" width="600"/>  
測量儀器，以此儀器蒐集受試者肌肉聲音  
<img src="https://github.com/Jason910315/Myotonus_Responses/blob/main/machine.jpg" width="600"/> 


繪製預測及實際值折線圖，比較各模型預測效能  

```python
def average_test_plot(pre,name):
    # 實際值
    true_array = [19.61,24.7,20.66,20.89,15.66,17.04,18.06,15.54,14.74,16.02]            
    average =  pd.DataFrame()
    true_df = pd.DataFrame(true_array)
  
    for i in range(0,10):
        index1 = i*40
        index2 = index1+40
        a = pre[index1:index2]
        k = a.mean()
        k = [k]
        k = pd.DataFrame(k)
        average = pd.concat([average,k],axis=0, ignore_index=True)
      
    
    plt.title(name+" Test")
    plt.scatter(average[0:5], true_df[0:5],c='b',alpha = .5)
    plt.scatter(average[5:10], true_df[5:10],c='r',alpha = .5)
    plt.xlabel('predict')
    plt.ylabel('true value')
    plt.axis([min(test_y), max(test_y)+0.2, min(test_y), max(test_y)+0.2])
    plt.legend((p1,p2),
           ('boy', 'girl'),
           scatterpoints=1,
           loc='lower right',
           ncol=5,
           fontsize=10)
    plt.savefig(path+'\\百分比\\迴歸圖(1s)\\'+name+'test(平均).png')
    plt.show()
        
    average=average.values
    true_df=true_df.values
    print(average)
    print(true_df)
        
    Regression = sum((average - np.mean(true_df))**2) # 回归平方和
    Residual   = sum((true_df - average)**2)         # 残差平方和    
    abs_reg=sum(abs(true_df - average))  

    mse =  Residual /10
    rmse = mse**0.5
    mae = abs_reg/10
    mape =(sum(((true_df - average)**2)/true_df))/10
    

    Residual   = sum((true_df - average)**2)          # 残差平方和
    total = sum((true_df-np.mean(true_df))**2) #总体平方和
    R_square   =  1-Residual / total  # 相关性系数R^2

    
    df = pd.DataFrame({
    'Model name': name,
    'dataset':'test',
    'MSE': np.round(mse,2),
    'RMSE':np.round(rmse,2),
    'MAE':np.round(mae,2),
    'MAPE':np.round(mape,2),
     'R-squared': np.round(R_square,2)},index=[0])
    
    global test_df
    test_df = pd.concat([test_df,df],axis=0, ignore_index=True)
    return test_df
```
建立 GradientBoosting 模型
```python
from sklearn.ensemble import GradientBoostingRegressor
GB = GradientBoostingRegressor(random_state=0,loss = 'huber',n_estimators=250)
GB.fit(train_X, train_y)

GB_predict_train = GB.predict(train_X) 
GB_predict_test = GB.predict(test_X) 
```
