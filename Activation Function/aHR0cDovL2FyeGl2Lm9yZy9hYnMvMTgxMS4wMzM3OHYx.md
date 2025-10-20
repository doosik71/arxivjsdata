# Activation Functions: Comparison of Trends in Practice and Research for Deep Learning

Chigozie Enyinna Nwankpa, Winifred Ijomah, Anthony Gachagan, and Stephen Marshall

## 🧩 Problem to Solve

이 논문은 딥러닝 애플리케이션에서 사용되는 활성화 함수(Activation Functions, AFs)에 대한 기존 연구를 체계적으로 조사하고, 최신 연구 결과와 실제 딥러닝 배포(특히 ImageNet 우승 아키텍처)에서의 활성화 함수 사용 경향을 비교하여 문헌상의 공백을 메우는 것을 목표로 합니다. 즉, 다양한 딥러닝 애플리케이션을 위해 가장 적절한 활성화 함수를 효과적으로 선택할 수 있도록 돕는 것이 주된 문제입니다.

## ✨ Key Contributions

- **활성화 함수(AFs)의 포괄적인 집대성:** 딥러닝에 활용되는 주요 활성화 함수와 그 변형들을 수학적 정의, 장점, 단점 및 적용 사례와 함께 상세히 정리했습니다.
- **실제 적용과 연구 간의 경향 비교 분석:** ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 우승 아키텍처 등 실제 배포 사례에서의 AF 사용 경향을 분석하고, 이를 최신 연구에서 제안된 AF들의 성능과 비교하여 실질적인 통찰력을 제공합니다.
- **AF 선택을 위한 의사결정 지원:** 각 함수의 특성과 실제 사용 패턴을 제시함으로써, 개발자들이 특정 딥러닝 애플리케이션에 가장 적합한 AF를 선택하는 데 필요한 정보를 제공합니다.
- **활성화 함수 연구의 현재와 미래 조명:** 소실/폭발 그래디언트 문제, 죽은 뉴런 문제 등 AF가 해결하고자 하는 주요 과제와 그 해결책으로서의 다양한 AF 발전 과정을 설명하고, 복합 활성화 함수와 같은 향후 연구 방향을 제시합니다.

## 📎 Related Works

이 논문은 딥러닝 분야의 다양한 선행 연구들을 참조합니다. 특히, LeNet5 [7], AlexNet [8], VGGNet [9], GoogleNet [10], ResNet [11] 등 딥러닝 아키텍처의 발전을 다루며, ImageNet 대회 우승 모델들을 실제 AF 사용 경향 분석의 근거로 활용합니다. 또한, 배치 정규화(Batch Normalization), 드롭아웃(Dropout), 적절한 가중치 초기화(Proper Initialization) [14], [18] 등 딥러닝 모델의 성능을 향상시키는 다른 기술들과 함께 활성화 함수의 중요성을 강조합니다. 초기 AF 연구로는 Elliott(1993) [47]의 연구와 Turian et al.(2009) [13]의 Softsign 함수 제안이 언급됩니다.

## 🛠️ Methodology

이 논문은 딥러닝 활성화 함수에 대한 **조사(Survey)** 방식으로 연구를 수행합니다.

1. **배경 및 활성화 함수 정의:** 딥러닝과 활성화 함수의 기본 개념, 역할(선형 출력을 비선형으로 변환), 그리고 그래디언트 문제(소실/폭발)와의 관계를 설명합니다.
2. **활성화 함수 유형별 분석:** Sigmoid, Hyperbolic Tangent (Tanh), Softmax, Softsign, Rectified Linear Unit (ReLU) 및 그 변형(Leaky ReLU, PReLU, RReLU, SReLU), Softplus, Exponential Linear Units (ELUs) 및 그 변형(PELU, SELU), Maxout, Swish, ELiSH 및 그 변형(HardELiSH) 등 총 21가지 활성화 함수를 개별적으로 소개합니다.
3. **수학적 정의 및 특성 제시:** 각 AF에 대해 수학적 수식(예: ReLU: $f(x) = \max(0, x)$), 주요 장점, 단점, 그리고 적절한 사용 시나리오를 설명합니다.
4. **실제 적용 경향 비교:** ImageNet Large Scale Visual Recognition Challenge (ILSVRC)의 역대 우승 아키텍처(AlexNet, ZFNet, GoogleNet, ResNet, SeNet 등)를 분석하여, 은닉층과 출력층에서 사용된 AF들의 유형과 빈도를 파악합니다.
5. **연구 결과와 실제 경향의 대조 및 논의:** 최신 연구에서 제안된 성능 개선 AF들과 실제 산업에서 주로 사용되는 AF들 간의 차이점을 논의하고, 이러한 차이가 발생하는 이유에 대한 통찰을 제공합니다.

## 📊 Results

- **활성화 함수 유형 정리:** Sigmoid, Tanh, Softmax, Softsign, ReLU, Softplus, ELU, Maxout, Swish, ELiSH 등 주요 AF와 이들의 변형들을 포함하여 총 21가지 활성화 함수의 상세한 정보를 제공했습니다.
- **ReLU와 Softmax의 지배력:** 대부분의 ImageNet 우승 아키텍처를 포함한 실제 딥러닝 애플리케이션에서 **ReLU가 은닉층 활성화 함수로 가장 널리 사용**되었으며, **Softmax는 다중 클래스 분류를 위한 출력층 활성화 함수로 보편적으로 활용**되었습니다 (표 I 참조).
- **최신 아키텍처에서의 Sigmoid 활용:** 2017년 ILSVRC 우승 아키텍처인 SeNet은 은닉층에 ReLU를 사용했지만, 출력층에는 Sigmoid 함수를 사용했습니다.
- **연구와 실제 적용 간의 격차:** 논문에서 소개된 Swish, ELiSH 등 많은 최신 연구 기반 AF들이 ReLU보다 더 나은 성능을 보고하지만, 실제 배포된 최첨단 딥러닝 아키텍처들은 여전히 ReLU와 Softmax/Sigmoid와 같은 검증된 함수들을 선호합니다. 이는 새로운 AF들이 아직 실용적 검증이나 광범위한 채택 단계에 이르지 못했음을 시사합니다.
- **ReLU 변형의 중요성:** Leaky ReLU (LReLU), Parametric ReLU (PReLU), Randomized Leaky ReLU (RReLU) 등 ReLU의 변형들은 표준 ReLU의 "죽은 뉴런" 문제 등을 해결하며, 분류 작업에서 더 나은 성능을 보였습니다.
- **매개변수화된(Parametric) AF의 부상:** PReLU, PELU, SReLU와 같이 학습 가능한 매개변수를 포함하는 AF들이 특정 환경에서 더 유연하고 우수한 성능을 나타냈습니다.

## 🧠 Insights & Discussion

- **실제 적용의 보수성:** 새로운 활성화 함수들이 연구적으로 뛰어난 성능을 보일지라도, 실제 딥러닝 배포에서는 안정성, 이해 용이성, 광범위한 검증 등의 이유로 ReLU와 Softmax와 같은 기존의 잘 확립된 함수들이 여전히 선호됩니다. 이는 연구 결과가 실제 산업에 적용되기까지 추가적인 검증과 시간이 필요함을 보여줍니다.
- **ReLU의 지속적인 중요성:** ReLU는 "죽은 뉴런" 문제와 같은 단점에도 불구하고, 계산 효율성과 소실 그래디언트 문제 완화 능력 덕분에 딥러닝의 핵심 활성화 함수로 자리매김했으며, 그 변형들은 이러한 단점을 보완하며 발전을 이어가고 있습니다.
- **학습 가능한 매개변수의 잠재력:** PReLU, PELU, SReLU와 같이 학습 가능한 매개변수를 가진 활성화 함수들은 모델이 데이터에 더 잘 적응하도록 돕는 유망한 방향을 제시하며, 향후 AF 연구의 중요한 초점이 될 것입니다.
- **복합 활성화 함수의 등장:** Swish, ELiSH와 같이 기존 AF들을 결합하여 새로운 특성을 가진 함수를 만드는 복합 AF는 정보 흐름 개선 및 그래디언트 문제 해결에 새로운 가능성을 열어줍니다.
- **주로 CNN 및 분류 작업에 집중:** 대부분의 활성화 함수 연구와 테스트가 컨볼루션 신경망(CNN)을 활용한 이미지 분류 작업에 집중되어 있다는 점은 딥러닝 연구의 주요 흐름을 반영하지만, 다른 도메인에서의 AF 적용에 대한 추가 연구의 필요성도 시사합니다.

## 📌 TL;DR

이 논문은 딥러닝 활성화 함수(AFs)의 포괄적인 개요를 제공하고, 최신 연구 결과와 실제 딥러닝 아키텍처(주로 ImageNet 우승 모델)에서의 AF 사용 경향을 비교 분석합니다. 핵심 문제는 다양한 딥러닝 애플리케이션에 적합한 AF를 선택하는 것이었습니다. 연구는 Sigmoid, Tanh, ReLU 및 그 변형, Softmax, ELU 등 21가지 AF의 수학적 정의와 특성을 다룹니다. 주요 결론은 **실제 딥러닝 애플리케이션에서는 ReLU가 은닉층에, Softmax가 출력층에 압도적으로 많이 사용된다는 것**입니다. 이는 연구에서 제안되는 더 성능 좋은 최신 AF들보다 검증된 AF들의 안정성과 실용성이 우선시됨을 보여줍니다. 논문은 또한 매개변수화된 AF와 복합 AF가 미래 연구의 중요한 방향임을 제시합니다.
