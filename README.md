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
3. epoch를 늘리고, call_back 추가 -> epoch를 늘려도 정확도의 향상을 체감하진 못함
4. 데이터 형태 변형 ((128,104) -> (128, 128) -> conv layer 추가 -> 미시도
5. hyper parameter 미세 조정 -> (3, 1, 1)이 최선이라고 판단. 
6. filter_num 조정하기? 어떻게? -> filter_num은 조정 결과 낮아지면 학습이 매우 느려짐.. 불가능하다 판단

다양한 방식으로 overfitting을 방지하기 위해 시도중이나
쉽지 않은것으로 판단
conv layer를 더 쌓고 싶지만, 데이터 형태 문제로 더이상 시도할 순 없을듯
