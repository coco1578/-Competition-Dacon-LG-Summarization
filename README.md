# Dacon LG Summarization 대회

### 최종 순위 Private 32 상위 7~8%
---

- 그 동안, Computer Vision 쪽 딥러닝 모델만 공부하다 처음 도전한 NLP 대회에서 개인적으로 나쁘지 않았다(?) 정도의 성적을 거둘 수 있어서 만족스러운 대회.

### 시도했던 내용

1. Bert 모델을 활용하여 입력 문장에서 중요 문장벡터를 추출 한 뒤 Decoder Layer에서 요약문을 생성하는 모델을 먼저 생각했었으나, NLP 관련 배경지식의 미흡으로 시도해보지 못함
2. 문장의 길이가 너무 길어 Tokenizer가 truncation을 할 수 밖에 없는 문장을 nested sentence로 변경해보는 방법도 고민해보았으면 좋았을 것 같음.
3. Score 측정 기준으로 Rouge1 Rouge2 RougeL을 사용했기 때문에 모델 파라미터의 save 기준을 Rouge score로 선정.
4. 맞춤법 관련 Post Processing을 해보았으나 오히려 스코어가 떨어지는 현상을 발견. (원인은 아직 찾지 못함)
5. WarmUP Scehduler나 Gradient Accumulation과 같은 방법도 잘 활용했으면 모델을 최적화하는데 도움이 됐을 것 같다는 아쉬움이 남음.
6. 추출 요약의 경우 앙상블 기법을 사용 할 수 있는데 생성 요약에는 이러한 기법이 없는지 궁금증이 남음.
