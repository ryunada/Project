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

<table>
  <tr>
    <td>
      <img src = "https://user-images.githubusercontent.com/87309905/209649054-39b84715-78a5-439f-bf6a-ee46e850958b.png">  
      epoch에 따른 train, val 데이터 셋의 accuracy
    </td>
    <td>
      <img src = "https://user-images.githubusercontent.com/87309905/209649223-6a6aad8a-edf8-4833-9edb-95dc9326e3ad.png">
      epoch에 따른 train, val 데이터셋의 loss
    </td>
  </tr>
</table>


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
  def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    # start_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df
 ```

I. 재료에 해당하는 이미지를 

II. 데이터 분석  

III. 이미지 인식 모델과 분석 모델을 이용하여 구현

IV. 웹 사이트 구현

