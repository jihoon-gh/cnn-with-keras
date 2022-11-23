### CNN with Keras

#### 현재 layer 구성
>
> VGG 16 기반
> 
> conv -> conv -> conv -> maxPooling 과정 3회
> 
> 그 후 dense(affine)
>

현재 정확도 약 58~62% 사이

정확도 올리기 위해 고려할만한 사항
1. DROPOUT LAYER 추가 (현재 반영)
2. 데이터 추가 (이미지 자체를 늘리기 OR 이미지를 변형하여 추가하기)
3. epoch를 늘리고, call_back 추가
4. 데이터 형태 변형 (정사각형 형태로) -> conv layer 추가