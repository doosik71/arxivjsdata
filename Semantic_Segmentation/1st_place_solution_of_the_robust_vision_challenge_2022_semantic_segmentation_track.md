# 1st Place Solution of The Robust Vision Challenge 2022 Semantic Segmentation Track

Junfei Xiao, Zhichao Xu, Shiyi Lan, Zhiding Yu, Alan Yuille, Anima Anandkumar (2022)

## 🧩 Problem to Solve

본 논문은 ECCV 2022에서 개최된 Robust Vision Challenge (RVC)의 Semantic Segmentation 트랙에서 우승한 솔루션을 제안한다. 해결하고자 하는 핵심 문제는 딥러닝 모델이 실제 환경(in the wild)에 배포되었을 때 발생하는 분포 변화(distributional shifts)와 자연적인 섭동(natural perturbations)에 취약한 문제, 즉 모델의 강건성(robustness) 부족이다.

특히 Semantic Segmentation 작업에서 단일 모델이 실내, 실외, 합성 데이터, 실제 데이터 등 서로 다른 도메인을 아우르는 다양한 벤치마크 데이터셋에서 일관되게 높은 성능을 내는 일반화(generalization) 능력을 확보하는 것이 본 연구의 목표이다.

## ✨ Key Contributions

본 솔루션의 중심 아이디어는 **강건성이 검증된 Vision Transformer (ViT) 기반의 백본과 대규모 다중 데이터셋 학습 전략을 결합**하는 것이다. 구체적으로는 다음과 같은 설계 방향을 가진다.

1. **Robust Backbone**: 이미지 분류 및 다운스트림 작업에서 높은 강건성을 보인 FAN-B-Hybrid 모델을 인코더로 채택하여 Out-of-distribution 시나리오에 대응한다.
2. **Efficient Framework**: 단순하면서도 효율적인 MLP 디코더를 사용하는 SegFormer 프레임워크를 통해 다중 레벨 특징을 융합한다.
3. **Large-scale Multi-dataset Training**: 서로 다른 성격의 9개 데이터셋을 통합하고, 데이터 불균형을 해소하는 단순한 리사이징 전략을 통해 도메인 일반화 성능을 극대화한다.

## 📎 Related Works

논문에서는 도메인 일반화(Domain Generalization)를 위해 도메인 랜덤화, 도메인 불변 표현 학습(domain invariant representation learning), disentanglement 학습, 메타 학습 등이 연구되어 왔음을 언급한다. 특히, 여러 데이터셋을 결합하고 레이블 공간을 정렬하여 학습시키는 Multi-dataset training 방식이 강력한 도메인 일반화 기법보다 더 좋은 성능을 낼 수 있음을 시사한 기존 연구(Mseg 등)를 참고하였다.

또한, 최근 Vision Transformers (ViTs)가 CNN 기반 모델보다 분포 외(out-of-distribution) 시나리오에서 훨씬 더 강건하다는 경향성에 주목하였다. 예를 들어 SegFormer는 자연적인 오염(corruption)이 포함된 Cityscapes-C 데이터셋에서 CNN 기반 방법론보다 우수한 결과를 보였으며, FAN(Fully Attentional Network)은 이미지 분류와 세그멘테이션 모두에서 최신 수준의 정확도와 강건성을 입증하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

본 모델은 **FAN-B-Hybrid**를 인코더(백본)로 사용하고, **SegFormer**를 세그멘테이션 프레임워크로 사용하는 구조이다.

1. **Backbone (Encoder)**: FAN-B-Hybrid 모델을 사용하며, ImageNet-22K에서 사전 학습된 가중치로 초기화한 후 ImageNet-1K에서 파인튜닝된 체크포인트를 사용한다.
2. **Segmentation Framework (Decoder)**: SegFormer의 MLP 디코더를 사용하여 초기 Convolution 블록, 마지막 FAN Transformer 블록, 그리고 최종 class attention 블록의 출력값인 다중 레벨 특징(multi-level features)을 융합하여 최종 세그멘테이션 마스크를 예측한다.

### 학습 데이터 및 전략

총 9개의 다양한 데이터셋(ADE20K, Cityscapes, Mapillary Vistas, ScanNet, VIPER, WildDash 2, IDD, BDD, COCO)을 통합하여 학습을 진행한다.

- **Unified Label Space**: 각 데이터셋의 서로 다른 레이블을 RVC 공식 저장소에서 제공하는 256개 클래스의 통합 레이블 공간(unified label space)으로 투영하여 학습한다.
- **Dataset Balancing**: 데이터셋 간의 크기 차이가 매우 크기 때문에(예: COCO가 WildDash 2보다 30배 이상 큼), 데이터 불균형을 완화하기 위해 다음과 같은 단순 리사이징 전략을 사용한다.
    $$\text{Repeat count} = 120,000 // \text{len(dataset)}$$
    즉, 각 데이터셋을 해당 수만큼 반복해서 샘플링하여 전체적인 데이터 균형을 맞춘다.

### 학습 목표 및 손실 함수

모델은 표준적인 교차 엔트로피 손실 함수(Cross-Entropy Loss)를 사용하여 학습된다. 별도의 손실 함수 가중치 조절이나 복잡한 하이퍼파라미터 튜닝 없이 학습을 진행하였다.

### 추론 및 후처리

- **Inference**: 테스트 시에는 Multi-scale testing(0.5, 0.75, 1.0, 1.25, 1.5, 1.75 배율)과 Flip 증강을 적용한다.
- **Post-processing**: 모델이 예측한 256개 클래스의 통합 레이블 맵을 각 벤치마크 데이터셋의 원래 레이블 공간으로 다시 투영하여 최종 결과를 산출한다.

## 📊 Results

### 실험 설정

- **측정 지표**: class mIoU (mean Intersection over Union)
- **비교 대상**: RVC 2020 우승 솔루션인 Mseg와 PyRX.
- **테스트 벤치마크**: ADE20K, Cityscapes, Mapillary, ScanNet, VIPER, WildDash 2 (총 6개 도메인).

### 정량적 결과

실험 결과, 제안 방법은 모든 벤치마크에서 이전 우승 솔루션들을 큰 차이로 제치고 1위를 기록하였다.

| Method | ADE20K | Cityscapes | Mapillary | ScanNet | VIPER | WildDash 2 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Mseg (2020/2nd) | 33.1 | 80.7 | 34.1 | 48.5 | 40.7 | 34.7 |
| PyRX (2020/1st) | 31.1 | 74.7 | 40.4 | 54.6 | 62.5 | 42.2 |
| **FAN-NV (Ours)** | **43.4** | **82.0** | **55.2** | **58.6** | **69.8** | **47.5** |

위 표에서 확인할 수 있듯이, 특히 Mapillary와 ADE20K와 같은 데이터셋에서 비약적인 성능 향상이 관찰되었다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 Vision Transformer 모델(특히 FAN)이 다중 데이터셋 학습과 결합되었을 때, 세그멘테이션 작업에서 매우 강력한 강건성과 일반화 능력을 보인다는 것을 입증하였다. 이는 ViT가 CNN보다 데이터의 전역적인 문맥을 더 잘 파악하고, 분포 변화에 덜 민감하다는 최근의 학술적 발견들을 실제 챌린지 환경에서 재확인한 결과라고 볼 수 있다.

### 한계 및 논의사항

논문에서는 다음과 같은 실무적 과제들을 언급하고 있다.

1. **연산 자원**: 데이터셋의 규모와 레이블 공간이 커짐에 따라 학습 시 필요한 계산량과 메모리 소비가 크게 증가한다. (실제로 64대의 V100 GPU를 사용하여 35시간 동안 학습함)
2. **배포 효율성**: 현재의 ViT 모델들은 성능은 뛰어나지만, 실제 장치에 배포했을 때의 추론 효율성(Efficiency) 문제가 여전히 해결해야 할 과제로 남아 있다.

## 📌 TL;DR

본 논문은 **FAN-B-Hybrid 백본**과 **SegFormer 프레임워크**를 결합하고, **9개의 대규모 다중 데이터셋**을 통합 학습시켜 RVC 2022 세그멘테이션 트랙에서 1위를 차지한 솔루션을 제시한다. 단순한 데이터 밸런싱 전략과 통합 레이블 공간을 활용하여 도메인 일반화 성능을 극대화하였으며, 이는 ViT 기반 모델이 다중 도메인 세그멘테이션 작업에서 CNN보다 뛰어난 강건성을 가짐을 보여준다. 향후 연구에서는 이러한 고성능 ViT 모델의 연산 효율성을 높이는 방향의 연구가 중요할 것으로 보인다.
