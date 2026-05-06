# Single- and Multi-Task Architectures for Surgical Workflow Challenge at M2CAI 2016

Andru P. Twinanda, Didier Mutter, Jacques Marescaux, Michel De Mathelin, and Nicolas Padoy (2016)

## 🧩 Problem to Solve

본 논문은 M2CAI 2016의 Surgical Workflow Challenge에서 제시된 과제인 담낭 절제술(cholecystectomy) 수술 영상 내의 8가지 수술 단계(surgical phases)를 식별하는 문제를 해결하고자 한다. 수술 단계 인식은 수술의 흐름을 모니터링하고 분석하는 데 있어 매우 중요한 과제이다.

본 연구의 목표는 딥러닝 아키텍처를 사용하여 수술 단계 인식 성능을 높이는 것이며, 특히 단일 작업(single-task) 네트워크와 다중 작업(multi-task) 네트워크의 성능을 비교하고, 수술 워크플로우의 시간적 제약 조건(temporal constraints)을 효율적으로 반영하기 위한 방법론을 탐색하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 두 가지 관점에서 제시된다. 첫째, 수술 단계 인식이라는 단일 작업뿐만 아니라 도구 존재 여부(tool presence detection)를 함께 학습하는 다중 작업 학습(multi-task learning)이 시각적 특징 추출의 변별력을 높여 인식 성능을 향상시킬 수 있다는 점이다. 둘째, 프레임 단위로 처리되는 CNN의 한계를 극복하기 위해, 기존의 Hidden Markov Model(HMM)보다 장기 의존성(long-term dependency)을 더 잘 학습할 수 있는 Long Short-Term Memory(LSTM) 네트워크를 도입하여 시간적 제약 조건을 강제하는 것이다.

## 📎 Related Works

논문은 저자들의 이전 연구([6])에서 제안된 PhaseNet과 EndoNet 아키텍처를 기반으로 한다. 기존의 수술 단계 인식 방식에서는 시간적 제약 조건을 반영하기 위해 HMM 기반의 접근 방식을 사용하였다. 그러나 HMM은 현재 상태가 오직 직전 상태에만 의존한다는 마르코프 가설(Markov assumption)에 기반하며, 시퀀스를 통해 전달되는 상태의 수가 클래스 수로 제한된다는 한계가 있다. 본 논문은 이러한 HMM의 제약을 극복하기 위해 LSTM을 대안으로 제시하며, 이를 통해 보다 유연한 시간적 모델링이 가능함을 보여주려 한다.

## 🛠️ Methodology

### 1. CNN 아키텍처 (특징 추출기)

본 연구에서는 ImageNet으로 사전 학습된 AlexNet을 백본으로 사용하여 두 가지 네트워크를 구성한다.

- **PhaseNet**: 수술 단계 인식만을 수행하는 단일 작업 네트워크이다.
- **EndoNet**: 수술 단계 인식과 도구 존재 여부 감지를 동시에 수행하는 다중 작업 네트워크이다.

두 네트워크 모두 AlexNet의 구조를 따르되, 최종 출력층에서 각각의 작업에 맞는 Fully Connected(FC) 레이어($fc_{phase}$, $fc_{tool}$)를 추가하여 파인튜닝(fine-tuning)한다.

### 2. 시간적 제약 조건 반영 파이프라인

CNN은 프레임 단위로 예측을 수행하므로, 수술의 순서라는 시간적 맥락을 반영하기 위해 다음의 두 가지 파이프라인을 제안한다.

#### (1) HMM 기반 파이프라인

- **절차**: CNN의 마지막 전 단계 레이어($fc7$ 또는 $fc8$)에서 이미지 특징을 추출한다 $\rightarrow$ Multi-class linear SVM을 통해 각 단계에 대한 신뢰도(confidence)를 계산한다 $\rightarrow$ 이 값을 Hierarchical HMM(HHMM)의 입력으로 사용하여 최종 예측을 수행한다.
- **특징**: 온라인 인식(online recognition)을 위해 forward 알고리즘을 사용하며, SVM의 출력 모델링을 위해 가우시안 혼합 모델(GMM)을 활용한다.

#### (2) LSTM 기반 파이프라인

- **절차**: CNN에서 추출된 이미지 특징을 1024개의 상태(states)를 가진 LSTM 네트워크에 입력한다 $\rightarrow$ LSTM의 출력을 8개의 노드(수술 단계 수)를 가진 FC 레이어로 전달하여 최종 신뢰도를 계산한다.
- **특징**: 메모리 제약으로 인해 CNN과 LSTM을 end-to-end로 학습시키지 않고, 특징 추출 후 LSTM을 별도로 학습시키는 분리 학습 방식을 채택하였다.

### 3. 손실 함수 및 평가 지표

수술 단계 인식의 성능 평가를 위해 Jaccard score를 사용하며, 수식은 다음과 같다.

$$J(GT, P) = \frac{GT \cap P}{GT \cup P}$$

여기서 $GT$는 Ground Truth(정답)를, $P$는 Prediction(예측)을 의미한다. 또한 일반적인 정확도(Accuracy) 지표를 함께 사용한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: M2CAI 2016 워크플로우 데이터셋(8단계)과 Cholec80 데이터셋(7단계, 도구 어노테이션 포함)을 사용하였다.
- **비교 대상**:
  - `PhaseNet-m2cai16`: M2CAI 데이터로 학습된 단일 작업 네트워크.
  - `PhaseNet-Cholec80`: Cholec80 데이터로 학습된 단일 작업 네트워크.
  - `EndoNet-Cholec80`: Cholec80 데이터로 학습된 다중 작업 네트워크.

### 2. 정량적 결과

실험 결과, 다음과 같은 경향성이 확인되었다.

- **다중 작업 학습의 효과**: HMM 파이프라인 사용 시, `EndoNet-Cholec80`이 `PhaseNet-m2cai16`보다 높은 성능(Jaccard $67.7\%$, Accuracy $80.6\%$)을 보였다. 이는 학습 데이터셋의 정의가 다름에도 불구하고, 다중 작업 학습을 통해 추출된 특징의 변별력이 더 뛰어나기 때문으로 분석된다.
- **LSTM의 성능**: 대체로 LSTM 기반 파이프라인이 HMM보다 우수한 성능을 보였다. 특히 `EndoNet-Cholec80`과 LSTM을 조합했을 때 Jaccard $69.8\%$, Accuracy $80.1\%$로 높은 성능을 기록하였다.
- **특이 사항**: `PhaseNet-m2cai16` 특징을 LSTM에 입력했을 때는 오히려 성능이 하락하는 모습이 관찰되었는데, 이는 LSTM의 하이퍼파라미터가 Cholec80 데이터셋에 최적화되어 있어 M2CAI 데이터셋과 맞지 않았기 때문으로 추정된다.

## 🧠 Insights & Discussion

본 논문은 수술 영상 분석에서 **다중 작업 학습(Multi-task Learning)**과 **시퀀스 모델링(LSTM)**의 결합이 강력한 성능을 낼 수 있음을 입증하였다. 특히 도구 감지라는 보조 작업(auxiliary task)을 함께 학습시키는 것이 주 작업인 단계 인식의 특징 표현력을 높여준다는 점은 시사하는 바가 크다.

또한, HMM의 마르코프 가설 한계를 LSTM으로 극복하여 수술의 시간적 흐름을 더 정확하게 모델링할 수 있음을 보여주었다. 다만, 본 연구에서는 메모리 문제로 인해 CNN과 LSTM을 분리하여 학습시켰으며, 이는 최적의 전역 해를 찾는 데 한계가 있을 수 있다. 향후 end-to-end 학습 구조를 구축한다면 더 높은 성능 향상이 가능할 것으로 보인다. 또한, 특정 네트워크에서 LSTM 성능이 하락한 점은 하이퍼파라미터 튜닝의 중요성을 시사한다.

## 📌 TL;DR

본 논문은 수술 단계 인식을 위해 다중 작업 CNN(EndoNet)과 LSTM 네트워크를 결합한 프레임워크를 제안하였다. 실험을 통해 도구 감지 작업을 병행 학습한 특징 추출기가 단일 작업 모델보다 우수하며, LSTM이 HMM보다 수술의 시간적 제약을 더 잘 반영함을 확인하였다. 이 결과는 향후 실시간 수술 모니터링 시스템 구축 및 정밀한 워크플로우 분석 연구에 중요한 기초 자료가 될 수 있다.
