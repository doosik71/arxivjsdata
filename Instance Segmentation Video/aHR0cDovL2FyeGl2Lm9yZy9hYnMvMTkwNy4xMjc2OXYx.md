# An Empirical Study of Propagation-based Methods for Video Object Segmentation

Hengkai Guo, Wenji Wang, Guanjun Guo, Huaxia Li, Jiachen Liu, Qian He, Xuefeng Xiao (2019)

## 🧩 Problem to Solve

본 논문은 비디오 객체 분할(Video Object Segmentation, VOS) 분야에서 널리 사용되는 전파 기반 방법론(propagation-based methods)들에 대한 체계적이고 공정한 비교 분석의 부재라는 문제를 해결하고자 한다. VOS는 첫 번째 프레임에서 주어진 마스크를 바탕으로 비디오의 모든 프레임에 대해 객체를 분할하는 작업이다. 이는 빠른 움직임, 외형의 급격한 변화, 가림 현상(occlusion) 및 유사한 방해 요소(distractors)로 인해 매우 도전적인 과제이다.

기존의 전파 기반 방법론들은 참조 프레임 정보, 임베딩 매칭, 동적 프레임 메모리 등 다양한 기법을 도입하며 성능을 향상시켜 왔으나, 각 논문마다 실험 설정이 상이하여 어떤 요소가 실제로 성능 향상에 기여하는지 객관적으로 비교하기 어려웠다. 따라서 본 연구의 목표는 전파 기반 VOS 방법론들을 통합된 관점에서 분석하고, 핵심 방법론, 입력 힌트(input cues), 다중 객체 결합 방식, 그리고 학습 전략이 최종 성능에 미치는 영향을 정밀하게 조사하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 전파 기반 VOS 시스템의 구성 요소를 체계적으로 분해하여 각 요소의 효용성을 검증한 실증적 연구(empirical study)에 있다. 연구진은 단순히 모델의 아키텍처뿐만 아니라, 입력으로 들어가는 정보의 종류(cues), 추론 과정에서의 확률값 처리 방식(soft probability), 다중 객체 마스크의 병합 전략, 그리고 오프라인 및 온라인 학습 패러다임이 성능에 미치는 영향을 심층적으로 분석하였다. 이러한 분석 결과를 바탕으로 기존의 Space-Time Memory (STM) 네트워크를 개선하여, DAVIS 2017 검증 세트에서 Global Mean 76.1이라는 높은 성능을 달성하였다.

## 📎 Related Works

논문에서는 전파 기반 VOS의 대표적인 세 가지 방법론인 MaskTrack, FEELVOS, 그리고 STM을 중점적으로 다룬다.

- **MaskTrack**: 정적 이미지에서 학습하여 비디오에 적용하는 초기 전파 기반 방식이다.
- **FEELVOS**: Fast End-to-End Embedding Learning을 통해 효율적인 임베딩 매칭을 수행한다.
- **STM (Space-Time Memory networks)**: 비디오 전체의 시공간적 정보를 메모리에 저장하고 Non-local matching을 통해 현재 프레임의 마스크를 예측하는 최신 기법이다.

기존 연구들은 각자 서로 다른 데이터 증강 기법이나 학습 전략을 사용했기 때문에, 본 논문은 이를 동일한 실험 환경(DeepLabv3+ 백본, 동일한 학습 스케줄 등) 아래에서 비교함으로써 기존 접근 방식들의 한계와 강점을 명확히 구분하였다.

## 🛠️ Methodology

본 논문은 전파 기반 VOS 시스템을 크게 세 가지 구성 요소로 정의하고 분석한다.

### 1. Feature Encoder (특징 추출기)
특징 추출기는 이미지와 마스크가 결합된 입력을 받아 마스크 디코더에 전달할 특징 맵을 생성한다.
- **Fusion Methods**: 서로 다른 프레임의 정보를 결합하기 위해 FEELVOS는 Correlation matching을, STM은 Non-local matching을 사용한다. 본 논문에서는 Correlation matching이 Non-local matching의 특수한 사례(특징 맵을 key로, 마스크를 value로 사용)로 볼 수 있음을 지적한다.
- **Input Cues**: 입력으로 사용하는 정보에 따라 성능이 달라지며, 이전 프레임의 마스크($M$), 이전 이미지($I$), 그리고 참조 프레임의 이미지 및 마스크($Ref$)를 조합하여 사용한다.

### 2. Mask Decoder (마스크 디코더)
디코더는 추출된 특징을 바탕으로 각 객체별 단일 마스크를 독립적으로 예측한 후, 이를 다중 객체 예측 결과로 병합한다.
- **Background Probabilities**: 배경 확률을 계산하는 방법으로 상수값(0.5) 사용, 확률 곱셈(probability production), 또는 배경 추적(mask tracking) 방식을 비교한다.
- **Normalization**: 병합 후 결과값을 정규화하는 방법으로 합계 기반 정규화(sum)와 로짓 집계(logit aggregation) 방식을 분석한다.

### 3. Training Paradigms (학습 패러다임)
학습은 크게 두 단계로 나뉜다.
- **Off-line Stage**: VOS 데이터셋이나 정적 이미지 데이터셋을 사용하여 모델을 사전 학습시킨다. 이때, 누적 오차를 반영하기 위해 시간적 역전파(Back-propagation Through Time, BPTT)를 적용하기도 한다.
- **On-line Stage**: 테스트 세트의 첫 번째 프레임 주석(annotation)을 사용하여 모델을 미세 조정(fine-tuning)한다. 여기서 데이터셋 단위(per-dataset) 혹은 비디오 단위(per-video)의 미세 조정 전략을 탐구한다.

### 4. 구현 세부 사항
- **Architecture**: DeepLabv3+ (ResNet-50, output stride 16)를 사용한다.
- **Loss Function**: 교차 엔트로피 손실(Cross Entropy Loss)을 최소화하며, Adam 옵티마이저와 poly 학습률 정책을 사용한다.
- **Hyperparameters**: 입력 종류에 따라 학습률을 다르게 설정하였다. (결합 입력: $1e-5$, 마스크 매칭: $5e-4$, Non-local 특징 매칭: $5e-5$)

## 📊 Results

### 1. 입력 힌트 및 융합 방법의 영향
실험 결과, 더 많은 입력 힌트(M $\rightarrow$ +I $\rightarrow$ +Ref)를 사용할수록 성능이 일관되게 향상되었다. 특히 STM의 Non-local matching 방식이 다른 융합 방법보다 압도적인 성능을 보였는데, 이는 시공간 차원에서 정보를 효율적으로 융합하여 더 강력한 특징을 생성하기 때문이다.

### 2. Soft Probability 입력의 효과
테스트 시 이전 프레임의 이진 마스크 대신 소프트 확률값을 입력으로 넣었을 때, FEELVOS와 STM은 성능이 향상되었으나 MaskTrack은 오히려 성능이 저하되었다. 이는 BPTT 없이 학습된 MaskTrack이 원-핫(one-hot) 표현에 과적합되어 확률값 입력에 취약하기 때문으로 분석된다.

### 3. 다중 객체 결합 및 정규화
BPTT를 통한 재귀적 학습을 수행하지 않은 경우, 정규화(Normalization)를 적용한 방법들이 적용하지 않은 방법보다 성능이 낮게 나타났다. 배경 확률 계산 방식은 최종 성능에 결정적인 영향을 미치지 않았으며, 본 연구에서는 정규화 없는 상수 배경 확률 방식을 채택하였다.

### 4. 온라인 학습(Fine-tuning) 전략
데이터셋 단위의 미세 조정(per-dataset FT)이 모델의 정확도를 향상시켰으며, 베이스라인 모델이 약할수록 미세 조정으로 인한 이득이 컸다. 비디오 단위의 미세 조정(per-video FT)은 본 실험 설정(단순 데이터 증강 사용)에서는 데이터셋 단위보다 성능이 낮게 나타났다.

### 5. 최종 벤치마크 결과
DAVIS 2017 검증 세트에서 STM 모델에 데이터셋 단위 미세 조정을 적용한 결과, Global Mean $G=76.1$ ($J=73.5, F=78.8$)을 달성하였다. 이는 많은 기존 방법론들을 상회하는 결과이며, 오직 PReMVOS와 원본 STM 논문의 결과만이 이를 앞선다.

## 🧠 Insights & Discussion

본 연구를 통해 전파 기반 VOS에서 성능을 결정짓는 핵심 요소는 단순히 모델 구조뿐만 아니라, **어떤 입력 정보를 활용하고 이를 어떻게 융합하며, 어떤 학습 전략을 취하는가**에 있다는 점이 밝혀졌다. 특히 STM과 같은 메모리 네트워크 구조가 시공간 정보를 통합하는 데 매우 강력하며, 적절한 입력 힌트의 추가가 성능 향상에 필수적임을 확인하였다.

다만, 본 논문에서는 시간적 제약으로 인해 최종 모델에 BPTT를 완전히 적용하지 못했다. 실험 결과 BPTT가 MaskTrack과 STM의 성능을 대폭 향상시키는 것이 확인되었으므로, 이를 전면적으로 적용한다면 더 높은 성능을 얻을 수 있을 것이다. 또한, 미세 조정 시 사용한 단순 증강 기법이 Lucid augmentation과 같은 고도화된 기법보다 효율이 낮았을 가능성이 있어, 이에 대한 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 전파 기반 VOS 방법론들을 통합된 관점에서 분석한 실증적 연구이다. 분석 결과, **STM(Memory Network) 구조**에 **풍부한 입력 힌트(Reference frame 등)**를 추가하고, **정규화 없는 단순 병합 전략**과 **데이터셋 단위의 미세 조정**을 결합했을 때 최적의 성능이 나타남을 증명하였다. 이 연구는 VOS 모델 설계 시 고려해야 할 핵심 요소들의 가이드라인을 제시하며, 향후 메모리 기반 네트워크의 효율적인 학습 및 추론 전략 연구에 중요한 기초 자료가 될 것으로 보인다.