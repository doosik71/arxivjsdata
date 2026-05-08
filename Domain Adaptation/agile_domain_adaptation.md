# Agile Domain Adaptation

Jingjing Li, Mengmeng Jing, Yue Xie, Ke Lu and Zi Huang (2019)

## 🧩 Problem to Solve

본 논문은 레이블이 잘 지정된 Source Domain의 지식을 레이블이 없는 Target Domain으로 전이시키는 Domain Adaptation(DA) 문제에서 발생하는 효율성과 정확성 사이의 상충 관계를 해결하고자 한다.

기존의 Domain Adaptation 접근 방식들은 Target Domain의 샘플들이 가지는 '적응 난이도(degree of difficulty)'의 차이를 간과하고, 모든 샘플에 대해 동일한 네트워크 구조와 연산 과정을 적용하는 문제가 있다. 일반적으로 단순한 네트워크(shadow framework)는 속도는 빠르지만 정밀도가 떨어지고, 정교한 깊은 네트워크(deep framework)는 정확도는 높지만 연산 비용이 많이 든다.

따라서 본 연구의 목표는 샘플별 적응 난이도에 따라 최적의 프레임워크를 유연하게 적용함으로써, Domain Adaptation 작업에서 정확도(Accuracy)와 속도(Speed) 사이의 근본적인 모순을 해결하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 네트워크 중간에 여러 개의 **Early Exit(조기 종료 지점)**을 배치하여, 쉬운 샘플은 빠르게 분류하고 어려운 샘플에만 더 많은 연산 자원을 투입하는 **Agile Domain Adaptation** 패러다임을 제안하는 것이다.

주요 기여 사항은 다음과 같다:

1. **다중 출구 구조(Multiple Exits) 제안**: 적응 난이도에 따라 서로 다른 지점에서 출력을 내보냄으로써 연산 비용을 획기적으로 줄이는 새로운 학습 패러다임을 제시하였다.
2. **ADAN(Agile Domain Adaptation Networks) 구현**: 제안한 패러다임을 실제 네트워크로 구현하여 기존의 Deep DA 방법론보다 효율적이고 효과적임을 입증하였다.
3. **특징 추출 층의 딜레마 해결**: 초기 층의 일반적 특징(General features)은 쉬운 샘플 분류에 사용하고, 후기 층의 구체적 특징(Specific features)은 어려운 샘플 분류에 사용함으로써, 네트워크의 모든 층을 효율적으로 활용하는 방안을 제시하였다.

## 📎 Related Works

### 기존 연구 및 한계

1. **전통적 DA 방법론**: 분포 정렬(Distribution Alignment), 특징 증강(Feature Augmentation) 등에 집중하며, 특징 추출 과정보다는 전이 기술 자체에 초점을 맞춘다.
2. **딥러닝 기반 DA 방법론**: 특징 추출과 지식 전이를 End-to-End 아키텍처로 처리한다. 대표적으로 MMD(Maximum Mean Discrepancy)를 이용한 DAN, JAN이나 GAN 기반의 ADDA, CoGAN 등이 있다.
3. **조기 종료 관련 연구**: BranchyNet이나 HDC(Hard-aware Deeply Cascaded Embedding)와 같이 샘플 난이도에 따라 조기 종료하는 기법이 존재한다.

### 차별점

기존의 BranchyNet이나 HDC는 학습 데이터와 테스트 데이터가 동일한 분포를 가진다고 가정하는 일반적인 머신러닝 환경을 대상으로 한다. 반면, 본 논문의 ADAN은 **Source와 Target 도메인 간의 분포 차이(Distribution Shift)**가 존재하는 전이 학습 환경에서 이 조기 종료 메커니즘을 적용하고, 각 층에서의 도메인 간 간극을 줄이는 적응 층(Adaptation layers)을 통합했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

ADAN은 백본 네트워크(Backbone Network)를 따라 여러 개의 출구(Exit)가 배치된 구조를 가진다. 각 출구는 백본의 특정 층 이후에 위치하며, 추가적인 컨볼루션 층과 적응 층을 포함하여 최종 분류를 수행한다.

### 학습 목표 및 손실 함수

각 출구 $\epsilon (\epsilon = 1, \dots, m)$에 대해 다음과 같은 손실 함수 $L^{exit}_\epsilon$을 정의한다.

$$L^{exit}_\epsilon = L^{sup} + \lambda L^{tran}$$

여기서 $\lambda > 0$은 두 손실 간의 균형을 맞추는 파라미터이다.

1. **지도 학습 손실 ($L^{sup}$)**: 레이블이 있는 Source Domain 데이터에 대해 Cross-Entropy 손실을 적용한다.
   $$J(\hat{y}, y) = -\frac{1}{|C|} \sum_{c \in C} y_c \log \hat{y}_c$$
2. **전이 손실 ($L^{tran}$)**: Source와 Target 도메인의 분포를 일치시키기 위해 Multi-kernel MMD를 사용한다. 특히, 원본 데이터가 아닌 네트워크 층의 활성화 값(Activations)에 대해 MMD를 계산하여 도메인 간의 정렬을 수행한다.
   $$L^{tran} = \text{MMD on layer activations}$$

전체 네트워크는 모든 출구의 손실 함수를 가중 합산하여 End-to-End 방식으로 최적화한다.
$$L = \sum_{\epsilon=1}^{m} w_\epsilon L^{exit}_\epsilon$$
(본 논문에서는 단순화를 위해 모든 $w_\epsilon = 1$로 설정하였다.)

### 추론 절차 및 조기 종료 메커니즘

테스트 단계에서는 샘플이 순차적으로 출구를 거치며, 각 출구에서 **샘플 엔트로피(Sample Entropy)**를 통해 분류 확신도를 측정한다.

$$\text{En}(y) = -\sum_{c \in C} y_c \log y_c$$

- **절차**:
    1. 샘플 $x$가 첫 번째 출구 $\text{exit}_1$에서 예측값 $y$를 생성한다.
    2. $\text{En}(y)$를 계산하여 설정된 임계값 $T_\epsilon$과 비교한다.
    3. 만약 $\text{En}(y) \le T_\epsilon$이면, 확신도가 높다고 판단하여 즉시 결과를 반환하고 종료한다.
    4. 그렇지 않으면 다음 출구 $\text{exit}_{\epsilon+1}$로 진행한다.
    5. 마지막 출구 $\text{exit}_m$에 도달할 때까지 반복한다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - MNIST $\to$ USPS (손글씨 숫자 인식)
  - Office-31 (Amazon, DSLR, Webcam 세 가지 서브셋 간의 객체 분류)
- **기본 아키텍처**: LeNet-5 (숫자 인식), ResNet-50 (객체 분류)
- **비교 대상**: TCA, GFK, CORAL, DANN, DAN, JAN 등 최신 DA 방법론 및 ResNet/LeNet 베이스라인.

### 주요 결과

1. **정확도 향상**: MNIST $\to$ USPS 작업에서 ADAN은 $91.3\%$의 정확도를 기록하며 비교 대상 방법론들(DANN $85.1\%$, DAN $81.1\%$ 등)보다 뛰어난 성능을 보였다. Office-31 데이터셋에서도 JAN 등의 SOTA 방법론보다 높은 평균 정확도를 달성하였다.
2. **효율성 증대**: MNIST $\to$ USPS 실험에서 임계값 조절을 통해 정확도 손실 거의 없이 실행 시간을 $2.42\text{ms}$에서 $0.83\text{ms}$로 단축하여 **약 3배의 속도 향상**을 이루었다.
3. **정성적 분석**: 시각화 결과, Source 도메인과 매우 유사한 '쉬운 샘플'은 첫 번째 출구에서 종료되고, 각도가 매우 다르거나 형태가 복잡한 '어려운 샘플'은 최종 출구까지 진행됨을 확인하여, 모델이 실제로 적응 난이도를 인식하고 있음을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 딥러닝 모델의 층이 깊어질수록 특징이 '일반적(General)'에서 '구체적(Specific)'으로 변한다는 특성을 영리하게 이용하였다. 쉬운 샘플은 일반적인 특징만으로도 충분히 분류 가능하므로 초기 층에서 처리하고, 어려운 샘플은 정교한 특징이 필요하므로 깊은 층까지 보내는 구조를 통해 **정확도와 속도라는 두 마리 토끼를 잡았다**고 평가할 수 있다. 특히, 이는 자원이 제한된 모바일 기기(Edge)와 고성능 서버(Cloud) 간의 분산 배포 가능성을 시사한다.

### 한계 및 논의사항

- **데이터셋 민감도**: Office-31의 Amazon 데이터셋과 같이 매우 어려운 데이터의 경우, 단순히 출구를 추가하는 것만으로는 성능 향상이 제한적이었다. 저자들은 출구의 위치를 뒤로 옮기거나 조기 종료 비율을 조정하는 튜닝이 필요함을 언급하였다.
- **임계값 설정**: 성능과 속도의 트레이드-오프가 임계값 $T$에 의해 결정되는데, 이를 자동으로 최적화하는 방법론에 대해서는 구체적으로 다루지 않았다.

## 📌 TL;DR

본 논문은 Domain Adaptation에서 샘플마다 다른 적응 난이도를 고려하여, 네트워크 중간에 **다중 출구(Multiple Exits)**를 배치한 **ADAN**을 제안한다. 이를 통해 쉬운 샘플은 빠르게, 어려운 샘플은 정밀하게 처리함으로써 **SOTA 수준의 정확도를 유지하면서도 연산 속도를 최대 3배 이상 향상**시켰다. 이 연구는 실시간 시스템 및 모바일 환경에서의 효율적인 도메인 적응 모델 적용에 중요한 가능성을 제시한다.
