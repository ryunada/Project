<img src = "https://user-images.githubusercontent.com/87309905/208438057-d3c55319-dfb8-4c6f-a519-4def533414c8.png" width/>

> # 분석을 위한 레시피 추천 서비스
> >#### 진행 기간 : 2022/11/04 -> 2022/12/17
> > #### 발표 : 2022/12/17
> > #### Data_Link :
> > #### Notion : 
> > > 내용
> > > 전체 일정

```markdown
1인 가구가 증가하는 시대에 코로나로 인하여 배달이 증가하고 그에 따른 배달비에 부담을 많이 느낍니다.
이러한 부담을 줄이고자 데이터 분석에 따른 레시피 추천 프로그램을 시작하게 되었습니다.
사용자의 성별, 연령대, 재료, 요리 난이도, 조리 시간 등 다양한 조건을 고려하여 1인 가구를 위한 레시피 추천하는 프로그램 입니다.
이를 통하여 식비에 대한 부담을 줄여주고 직접 요리를 간평하게 할 수 있도록 도음을 주고자 합니다.
```
- 사용 데이터
  - 레시피 데이터 : 레시피명, 조리시간, 요리 종류, 재료, 요리 난이도
  - 이미지 데이터 : 재료에 해당하는 이미지
  - 레시피 검색량 데이터 : 레시피 명에 해당하는 검색량(키워드 사운드)
  - 성별, 연령별 검색량 데이터 : 레시피에 해당하는 성별, 연령별 검색량(네이터 트렌드)
  - 기상청 데이터 : 온도, 습도, 강수량, 운량

## 💻 프로젝트 진행 (이미지 인식 and 데이터 분석)

### I. 이미지 인식  
> 1. 데이터 수집
> https://kadx.co.kr/product/detail/0c5ec800-4fc2-11eb-8b6e-e776ccea3964
> KADX 농식품 빅데이터 거래소에 있는 만개의 레시피 데이터를 활용

> 2. 데이터 전처리 
>   
> I. 레시피 데이터 중 재료 데이터를 추출  
> II. 중복 제거 후 리스트로 추출  

> 3. 이미지 데이터 수집  
>  
> I. 재료에 해당하는 이미지를 네이버 & 구글을 통하여 재료당 100개씩 추출 (셀레니움 크롤링 사용)  
> II. 불필요한 배경이나 사물들이 포함된 이미지 육안으로 제거  
> III. 이미지에 해당하는 라벨링 작업을 시행  
> IV. Colab으로 이미지 인식 실행  
> 
> <table>
>   <tr>
>     <td>
>       <img src = "https://user-images.githubusercontent.com/87309905/209649054-39b84715-78a5-439f-bf6a-ee46e850958b.png">  
>       epoch에 따른 train, val 데이터 셋의 accuracy
>     </td>
>     <td>
>       <img src = "https://user-images.githubusercontent.com/87309905/209649223-6a6aad8a-edf8-4833-9edb-95dc9326e3ad.png">
>       epoch에 따른 train, val 데이터셋의 loss
>     </td>
>   </tr>
> </table>


```
# ...
# Epoch 146/300
# 100/100 [==============================] - 6s 63ms/step - loss: 0.5831 - acc: 0.8233 - val_loss: 0.9350 - val_acc: 0.7207
# Epoch 147/300
# 100/100 [==============================] - 6s 62ms/step - loss: 0.5959 - acc: 0.8100 - val_loss: 0.9028 - val_acc: 0.7307
# Epoch 148/300
# 100/100 [==============================] - 6s 62ms/step - loss: 0.6300 - acc: 0.8053 - val_loss: 0.8664 - val_acc: 0.7260`

# 모델 평가
print("-- Evaluate --")
scores = model.evaluate(test_generator)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))
# acc: 73.29%
```

> 결과는 약 73%의 정확도를 보였습니다.  
> 정확도를 높이기 위하여 모델을 수정 하였습니다.  
> 
> YOLOv4 커스텀 학습을 통하여 진행하였지만 Colab에서 제공하는 GPU를 초과하여  
> YOLOv4 -> YOLOv4_tiny를 사용  
![Untitled (2)](https://user-images.githubusercontent.com/87309905/209650438-9b8115da-2808-40b9-8bd3-41a145bd90ad.png)
> YOLOv4_tiny 학습 결과
> - 학습 결과 대부분의 재료를 잘 인식 하였습니다.
> - 이 모델을 재료 이미지 인식 모델로 사용합니다.

### II) 데이터 분석
> I. 데이터 수집  
>   
> https://kadx.co.kr/product/detail/0c5ec800-4fc2-11eb-8b6e-e776ccea3964
> KADX 농식품 빅데이터 거래소에 있는 만개의 레시피 데이터를 활용  
>   
> https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36  
> 기상청에 있는 기온, 습도, 강수량, 운량 데이터 활용  
> 
> https://keywordsound.com/  
> 키워드 사운드를 통하여 검색량 데이터 수집. 
> > 과도하게 크롤링을 하여 서버를 마비 시켰습니다. 개인적으로 연락을 드려 사과를 드렸고 좋게 마무리 되었습니다.
> > 크롤링은 항상 신중하고 조심히 해야 하는 것을 배웠습니다.
> > 
> ><img width="669" alt="스크린샷 2022-12-24 오후 8 51 31" src="https://user-images.githubusercontent.com/87309905/209650976-bf31381b-e111-48f2-9827-b1a28c8a6e39.png">
> https://datalab.naver.com/keyword/trendSearch.naver  
> 네이버 트렌드를 이용하여 성별, 연령별 데이터의 비율 수집  

> II. 데이터 전처리  
>  
> 기준 : 날짜 2021년 1/1 ~ 12/31  
> 만개의 레시피 : 레시피 명  
> 키워드 사운드 : 레시피명에 해당하는 검색량  
> 키워드 사운드 + 네이버 트랜드 :  
> -> max(레시피명에 해당하는 검색량) * 네이버 트랜드 비율(성별, 연령 생각해야함)  
> 기상청 데이터 : 기온, 습도, 강수량, 운량  
>  
>  모든 데이터를 날짜별로 정리를 하여 하나의 데이터로 만들었습니다.  

> III. 데이터 분석   
> 날씨, 요일, 계절 등 정보와 사용자가 입력하는 재료, 나이, 성별에 따라 만들 수 있는 요리의 레시피 중, 검색량이 많을 것 같은 레시피를 예측하기 위한 모  델을 얻고자 하였습니다.  
>   
> - 데이터 분석을 할 때 메모리 사용량을 줄이기 위하여 DataFrame의 DataType을 바꾸어 주었습니다. 덕분에 메모리 error없이 학습 진행
> ```python
>  def reduce_mem_usage(df):
>    # start_mem = df.memory_usage().sum() / 1024**2
>    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
>
>    for col in df.columns:
>        col_type = df[col].dtype
>
>        if col_type != object:
>            c_min = df[col].min()
>            c_max = df[col].max()
>            if str(col_type)[:3] == 'int':
>                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
>                    df[col] = df[col].astype(np.int8)
>                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
>                    df[col] = df[col].astype(np.int16)
>                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
>                    df[col] = df[col].astype(np.int32)
>                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
>                    df[col] = df[col].astype(np.int64)
>            else:
>                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
>                    df[col] = df[col].astype(np.float16)
>                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
>                    df[col] = df[col].astype(np.float32)
>                else:
>                    df[col] = df[col].astype(np.float64)
>
>    return df
> ```
> > - 수집한 데이터를 Ensemble학습 중 LightGBM을 사용
> >   
> > LightGBM
> > ```python  
> > from lightgbm import LGBMRegressor
> >  
> > lgbm_t = LGBMRegressor(random_state=1, learning_rate=0.01, n_estimators=2000, colsample_bytree=0.9, subsample=0.7, max_depth=5)
> >  
> > lgbm_t.fit(x_train, y_train). 
> > lgbm_t_pred = lgbm_t.predict(x_test)
> > 
> > print('LGBMRegressor')
> > print('MAE:', mean_absolute_error(y_test, lgbm_t_pred)) # (평균 절대 오차) 예측값과 실제값의 차이의 절대값에 대한 평균
> > print('MSE:', mean_squared_error(y_test, lgbm_t_pred)) # (평균 제곱 오차) 예측값과 실제값의 차이의 제곱에 대한 평균
> > print('RMSE:', np.sqrt(mean_squared_error(y_test, lgbm_t_pred))) # (평균 제곱근 오차) 예측값과 실제값의 차이의 제곱에 대한 평균의 제곱근
> > print('R2:', r2_score(y_test, lgbm_t_pred)) # (결정 계수) 1에 가까울수록 예측값과 실제값이 가깝다는 의미
> >  
> > # 예측값과 실제값 비교 csv 파일로 저장
> > df2 = pd.DataFrame({'Actual': y_test, 'Predicted': lgbm_t_pred})
> > df2.to_csv(os.path.join(path, 'result_lgb.csv'), index=True)
> > 
> > # 모델 저장
> > import pickle
> > 
> > with open(os.path.join(path, 'lgbm_t.pkl'), 'wb') as f:
> >     pickle.dump(lgbm_t, f)
> >       
> > # # 모델 불러오기
> > # with open(os.path.join(path, 'lgbm_t.pkl'), 'rb') as f:
> > #     lgbm_t = pickle.load(f)
> > 
>  
> > GridSearchCV를 통하여 최적의 Parameter를 찾아, 찾은 값으로 학습을 진행
> > ```python
> > # ligthgbm 최적의 parameter 찾기
> > from sklearn.model_selection import GridSearchCV
> > 
> > lgbm = LGBMRegressor(random_state=1)
> >
> > params = {
> >     'max_depth': [15, 18, 19, 21, 23],
> >     'num_leaves': [50, 80, 100, 120, 150],
> >     'min_data_in_leaf': [100],
> >     'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
> >     'colsample_bytree': [0.5, 0.7, 0.9, 1],
> >     'subsample': [0.5, 0.7, 0.9, 1]
> > }
> > 
> > grid_cv = GridSearchCV(lgbm, param_grid=params, cv=5, n_jobs=-1)
> > grid_cv.fit(x_train, y_train, verbose=1, eval_metric=['rmse', 'mae'], eval_set=[(x_train, y_train), (x_test, y_test)], early_stopping_rounds=100)
> > 
> > print('최적의 파라미터:', grid_cv.best_params_)
> > print('최고 예측 정확도: {0:.4f}'.format(grid_cv.best_score_))
> > ```
> > 
> > ```python
> >lgbm_t = LGBMRegressor(random_state=1, n_estimators=2000, max_depth=15, num_leaves=150, min_data_in_leaf=100, learning_rate=0.2,
> >                        colsample_bytree=1, subsample=0.7)
> > ```
> > Parameter를 튜닝한 결과 2%정도의 더 높은 예측 정확도를 보였습니다.
> > 
> > <table> 
> >   <tr>
> >     <td>
> >       <img src="https://user-images.githubusercontent.com/87309905/209656040-9b1b55d3-1793-4b9b-9b30-da406b38a34f.png"> Parameter튜닝 전 기본값으로 학습
> >     </td>
> >     <td>
> >       <img src="https://user-images.githubusercontent.com/87309905/209656432-e0b83533-399b-4187-b1f7-d3584fc5d6ad.png"> Parameter 튜닝 후 얻은 결과로 학습
> >     </td>
> >   </tr>
> > </table>
> > - 결과적으로 완벽히 일치하지는 않으나, Predicted값이 Actual값과 비슷한 흐름을 가진다는 것을 알 수 있습니다.  
> > - 즉, 검색량을 정확히 예측을 할 수 없으나 상대적인 변화를 에측할 수 있습니다.  
> > - 더 많은 feature와 데이터들이 있었더라면 더욱 정확한 예측값을 얻을 수 있지 않을까라는 아쉬움이 남았습니다.  
> >     
> > 분석한 모델을 저장. 
> > ```python
> > import pickle
> > with open(os.path.join(path, 'lgbm_t.pkl'), 'wb') as f:
> >    pickle.dump(lgbm_t.fit(x_train, y_train), f)
> > ```

> 이미지 인식 모델과 분석 모델을 활용한 레시피 추천  
>  
> - 재료 이미지를 인식하여 재료를 파별하여 만들 수 있는 레시피 먼저 제공  
> - 제공된 레시피 중 성별, 나이, 날씨 등 다양한 조건을 조합여 최종적으로 검색량이 가장 높은, 즉, 사람들이 가장 즐겨 먹는 레시피를 학습 모델을 통하여 알려줍니다.  
> ```python
> def yolo(frame, size, score_threshold, nms_threshold):
>     """YOLO 시작"""
>     # YOLO 네트워크 불러오기
>     net = cv2.dnn.readNet("weights 파일 경로",
>                           "cfg 파일 경로")
>     layer_names = net.getLayerNames()
>     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
> 
>     # 클래스의 갯수만큼 랜덤 RGB 배열을 생성
>     colors = np.random.uniform(0, 255, size=(len(classes), 3))
>
>     # 이미지의 높이, 너비, 채널 받아오기
>     height, width, channels = frame.shape
> 
>     # 네트워크에 넣기 위한 전처리
>     blob = cv2.dnn.blobFromImage(frame, 0.00392, (size, size), (0, 0, 0), True, crop=False)
> 
>     # 전처리된 blob 네트워크에 입력
>     net.setInput(blob)
> 
>     # 결과 받아오기
>     outs = net.forward(output_layers)
> 
>     # 각각의 데이터를 저장할 빈 리스트
>     class_ids = []
>     confidences = []
>     boxes = []
> 
>     for out in outs:
>         for detection in out:
>             scores = detection[5:]
>             class_id = np.argmax(scores)
>             confidence = scores[class_id]
> 
>             if confidence > 0.1:
>                 # 탐지된 객체의 너비, 높이 및 중앙 좌표값 찾기
>                 center_x = int(detection[0] * width)
>                 center_y = int(detection[1] * height)
>                 w = int(detection[2] * width)
>                 h = int(detection[3] * height)
> 
>                 # 객체의 사각형 테두리 중 좌상단 좌표값 찾기
>                 x = int(center_x - w / 2)
>                 y = int(center_y - h / 2)
> 
>                 boxes.append([x, y, w, h])
>                 confidences.append(float(confidence))
>                 class_ids.append(class_id)
> 
>     # 후보 박스(x, y, width, height)와 confidence(상자가 물체일 확률) 출력
>     print(f"boxes: {boxes}")
>     print(f"confidences: {confidences}")
> 
>     # Non Maximum Suppression (겹쳐있는 박스 중 confidence 가 가장 높은 박스를 선택)
>     indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=score_threshold, nms_threshold=nms_threshold)
> 
>     # 후보 박스 중 선택된 박스의 인덱스 출력
>     print(f"indexes: ", end='')
>     for index in indexes:
>         print(index, end=' ')
>     print("\n\n============================== classes ==============================")
> 
>     for i in range(len(boxes)):
>         if i in indexes:
>             x, y, w, h = boxes[i]
>             class_name = classes[class_ids[i]]
>             label = f"{class_name} {confidences[i]:.2f}"
>             color = colors[class_ids[i]]
> 
>             # 사각형 테두리 그리기 및 텍스트 쓰기
>             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
>             cv2.rectangle(frame, (x - 1, y), (x + len(class_name) * 13 + 65, y - 25), color, -1)
>             cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
> 
>             # 탐지된 객체의 정보 출력
>             print(f"[{class_name}({i})] conf: {confidences[i]} / x: {x} / y: {y} / width: {w} / height: {h}")
>             globals()['ingre_list'].append(class_name)
> 
>     return frame
> ```
> -> YOLOv4-tiny로 학습된 모델로 이미지를 인식하고 인식된 이미지와 인식된 재료의 결과를 출력
> 
> ```python
> # lgbm_t.pkl 파일 불러오기
> with open('학습된 lgbm pickle 파일', 'rb') as f:
>     lgbm_t = pickle.load(f)
> 
> # print(lgbm_t)
> 
> # 임시로 넣은 조건
> Gender = 1 # 남자
> Age = 20 # 20대
> Temperature = 20.0 # 온도
> Precipitation = 0.0 # 강수량
> Humidity = 0.0 # 습도
> Cloud = 0.0 # 구름
> Month = 12 # 12월
> Season = 4 # 겨울
> Weekday = 5 # 토요일
> 
> # print(df2)
>
> model = lgbm_t
> 
> df2 = df2['CKG_NM']
> df2 = df2.values.reshape(-1, 1)
> # print(df2)
> 
> df3 = pd.DataFrame(columns=['RCP_NM', 'score'])
> 
> for i in range(len(df2)):
>     input_data = [df2[i][0], Gender, Age, Temperature, Precipitation, Humidity, Cloud, Month, Season, Weekday]
>     # print(input_data)
>     # print(model.predict([input_data]))
>     # df3에 df2의 CKG_NM과 model.predict([input_data])의 예측값을 전부 저장
>     df3 = df3.append({'RCP_NM': df2[i][0], 'score': model.predict([input_data])}, ignore_index=True)
> 
> # df3의 RCP_NM이 df1의 CKG_NM2가 같으면 CKG_NM을 RCP_NM으로 저장
> df3['RCP_NM'] = df3['RCP_NM'].map(df1.set_index('CKG_NM2')['CKG_NM'])
> # df3의 score를 내림차순으로 정렬
> df3 = df3.sort_values(by='score', ascending=False)
> print(df3.head())
> ```
> > LightGBM으로 학습된 모델을 사용하여 인식된 재료와, 미리 입력된 성별, 날씨 계절 등 다양한 조건을 조합하여 최종적으로 검색량이 높은 레시피를 예측하여 추천
> 예측 결과
> 조건 : 12월, 겨울, 토요일, 강수X, 구름X, 습도X, 20대, 남자
> 재료
> ![Untitled (5)](https://user-images.githubusercontent.com/87309905/209658913-5075c3b8-32d3-4126-8864-cef879af47dc.png)
> 복숭아(계란), 대파, 마늘, 스팸
>  
> > 결과  
> > RCP_NM                score
> > 292 떡볶이 [135.63899760197575] 
> > 291 떡꼬치 [135.63899760197575] 
> > 290 떡갈비꼬치 [135.63899760197575] 
> > 289 떡갈비구이 [135.63899760197575] 
> > 472 샤브샤브 [132.77751317420473]
>  
> Score가 높은 순으로 1위에서 5위입니다.  
> 떡볶이가 가장 높다고 예측. 
> 
> -  계란이 왜 복숭아로 인식되었는지, 다시 살펴봐야 할것 같습니다.  
> - 그 이외에는 잘 인식되는 것으로보아, YOLOv4-tiny를 잘 사용한것 같습니다.  

> IV. 웹페이지 구현
>
> 이미지 인식을 구현하였으나 웹페이지에 적용을 시키지는 못하여 아쉬움이 있었습니다.

> IV. 어플리케이션 화면 구성
>

## 배운점
- 크롤링은 신중하고 조심히 다뤄야 하는 것을 배웠습니다.
'키워드 사운드'라는 홈페이지를 Selenium으로 크롤링을 하다가 많은 트래픽이 몰려 사이트가 마비되었습니다.
이후 사이트 개발자님에게 피해를 입혀, 개인적으로 연락드려 사과드렸습니다.
