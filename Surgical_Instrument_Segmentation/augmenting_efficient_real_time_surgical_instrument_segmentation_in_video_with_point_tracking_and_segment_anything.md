# Augmenting Efficient Real-time Surgical Instrument Segmentation in Video with Point Tracking and Segment Anything

Zijian Wu, Adam Schmidt, Peter Kazanzides, and Septimiu E. Salcudean (2024)

## 🧩 Problem to Solve

본 논문은 로봇 보조 수술(Robotically Assisted Surgery) 환경에서 수술 도구 분할(Surgical Instrument Segmentation, SIS)을 실시간으로 수행하는 것을 목표로 한다. 수술 도구 분할은 증강 현실(AR) 가이드나 수술 장면 이해와 같은 후속 응용 프로그램에 필수적인 시각적 단서를 제공하는 기초적인 작업이다.

그러나 수술 영상에서의 분할은 다음과 같은 이유로 매우 도전적이다.

1. **환경적 요인**: 가려짐(occlusion), 혈액, 연기, 모션 아티팩트 및 조명 변화가 빈번하게 발생한다.
2. **데이터 부족**: 의료 영상의 특성상 고품질의 어노테이션을 생성하는 과정이 매우 노동 집약적이며 전문 지식을 요구하므로 대규모 데이터셋 확보가 어렵다.
3. **기존 모델의 한계**: 최근 등장한 Segment Anything Model (SAM)은 강력한 제로샷(zero-shot) 일반화 능력을 갖추고 있으나, 거대한 이미지 인코더 구조로 인해 계산 비용이 매우 높아 실시간 추론이 불가능하며, 의료 영상 도메인에서의 성능 저하 문제가 보고되고 있다.

따라서 본 연구는 임상 적용이 가능하도록 **추론 효율성(실시간성)**을 확보하면서도, **수술 장면에서의 일반화 성능**을 높인 비디오 수술 도구 분할 프레임워크를 개발하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 비디오 객체 분할(Video Object Segmentation, VOS) 문제를 **'이미지 수준의 분할(Image-level Segmentation)'**과 **'범용적 시간적 전파(Universal Temporal Propagation)'**로 분리(decouple)하는 것이다.

주요 기여 사항은 다음과 같다.

1. **실시간 비디오 SIS 프레임워크 제안**: 온라인 포인트 트래커(Online Point Tracker)와 수술 도구 분할에 최적화되어 파인튜닝된 경량 SAM 모델을 결합하여 높은 효율성과 정확도를 동시에 달성하였다.
2. **경량 SAM의 포인트 프롬프트 기반 파인튜닝 전략**: MobileSAM을 수술 데이터셋으로 파인튜닝하여 경량 네트워크의 성능 저하 문제를 해결하였으며, 소수의 데이터셋만으로도 학습되지 않은 수술 영상에 대해 우수한 일반화 성능을 보임을 입증하였다.

## 📎 Related Works

### Tracking Any Point (TAP)

TAP의 목적은 비디오 전체에서 임의의 물리적 포인트의 움직임을 추정하는 것이다. TAP-Vid, PIPs++, TAPIR 등이 제안되었으며, 특히 CoTracker는 쿼리 포인트 세트를 공동으로 추적함으로써 최신 성능을 보여준다. 기존의 광학 흐름(Optical flow) 방식은 시간이 지남에 따라 오차가 누적되고 가려짐 현상에 취약하지만, 최신 TAP 모델들은 이러한 문제에 대해 강건한 모습을 보인다.

### Segment Anything Model (SAM)

SAM은 10억 개의 마스크로 학습된 강력한 파운데이션 모델이지만, 의료 및 수술 영상과 같은 특수 도메인에서는 도메인 갭(domain gap)으로 인해 성능이 떨어진다. 이를 해결하기 위해 SurgicalSAM, AdaptiveSAM 등이 제안되었으나, 이들은 여전히 실시간 추론 속도를 확보하지 못했다. 또한, FastSAM이나 MobileSAM 같은 경량화 모델들이 제안되었으나 수술 장면의 특수성(반사광, 혈액 등)에 대응하기 위한 추가적인 튜닝이 필요하다.

## 🛠️ Methodology

### 전체 파이프라인

본 프레임워크는 크게 **포인트 트래커(Point Tracker)**와 **포인트 기반 분할 모델(Point-based Segmentation Model)**의 두 가지 구성 요소로 이루어진다. 전체 흐름은 다음과 같다.

1. 첫 번째 프레임에서 관심 영역(ROI)을 지정하여 초기 마스크를 생성한다.
2. 생성된 마스크 내에서 샘플링 전략을 통해 쿼리 포인트 세트를 초기화한다.
3. 포인트 트래커를 통해 이 포인트들을 비디오 시퀀스 전체에서 추적한다.
4. 추적된 포인트들을 매 프레임 SAM의 프롬프트로 입력하여 최종 마스크를 생성한다.

### 상세 구성 요소 및 절차

#### 1. 전처리 및 쿼리 포인트 초기화

초기 마스크 생성에는 SAM(수동 프롬프트 입력) 또는 CLIPSeg(텍스트 프롬프트 "surgical tool" 입력)를 사용한다. 생성된 마스크 내에서 포인트들을 효율적으로 추출하기 위해 **K-Medoids clustering**을 사용하며, 각 인스턴스당 5개의 중심점을 선택한다. 초기 포인트 세트를 다음과 같이 정의한다.
$$P_0 = \{(p_i, t_0)\}, \quad p_i = (x_i, y_i), \quad i = 1, \dots, N$$

#### 2. TAP + SAM 프레임워크

비디오 $V=\{I_t\}$가 주어졌을 때, 포인트 트래커(TAP)를 통해 시간 $t$에서의 쿼리 포인트 위치 $P_t$를 예측한다.
$$P_t = \text{TAP}(V, P_0)$$
예측된 $P_t$와 현재 이미지 $I_t$를 분할 모델(Seg)에 입력하여 마스크 $M_t$를 생성한다.
$$M_t = \text{Seg}(I_t, P_t)$$
본 연구에서는 TAP 모델로 **CoTracker**를, 분할 모델로는 파인튜닝된 **MobileSAM**을 사용한다.

#### 3. MobileSAM 파인튜닝 전략

경량 모델의 성능 저하를 막기 위해 MobileSAM의 **이미지 인코더(Image Encoder)**와 **마스크 디코더(Mask Decoder)**를 모두 업데이트하는 Full Fine-tuning을 수행한다. (프롬프트 인코더는 동결)

- **학습 데이터**: 인스턴스 수준 라벨의 경우 객체 내부에서 5개 포인트를 무작위 샘플링하여 프롬프트로 사용한다.
- **손실 함수**: Binary Cross Entropy (BCE) 손실과 Dice 손실을 가중치 없이 결합하여 사용한다.
$$L = L_{BCE} + L_{Dice}$$
- **최적화**: AdamW 옵티마이저, 코사인 감쇠(cosine decay) 스케줄러를 사용하며 50 epoch 동안 학습시킨다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis 2015, UCL dVRK, CholecSeg8k (정량 평가), ROBUST-MIS 2019, STIR (정성 평가).
- **비교 대상**: XMem (반지도 학습 기반 VOS), TransUNet, SwinUNet (완전 지도 학습 기반 이미지 분할).
- **지표**: IoU (Intersection over Union), Dice Coefficient.

### 정량적 결과

- **EndoVis 2015**: $\text{IoU} = 84.4$, $\text{Dice} = 91.0$를 기록하며 SOTA 방법인 XMem(82.6 IoU)을 능가하였다.
- **UCL dVRK**: $\text{IoU} = 89.4$, $\text{Dice} = 93.8$로 XMem(91.9 IoU)보다는 약간 낮으나 경쟁력 있는 성능을 보였다.
- **CholecSeg8k**: $\text{IoU} = 81.9$, $\text{Dice} = 88.6$를 달성하여 SwinUNet과 대등한 성능을 보였다.

### 효율성 평가

- **추론 속도**: NVIDIA RTX 4090 GPU에서 **90 FPS** (11ms), RTX 4060 GPU에서 **26 FPS** (38ms)를 달성하여 실시간 처리가 가능함을 입증하였다.
- **자원 소모**: 추론 메모리는 2.8G, 학습 가능한 파라미터 수는 10.1M으로 매우 효율적이다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **시간적 일관성**: 범용 TAP(CoTracker)를 사용함으로써 XMem과 같은 전파 기반 모델보다 긴 비디오 시퀀스에서 더 강건한 성능을 보였다.
- **유연성**: XMem은 첫 프레임의 정교한 Ground Truth 마스크가 필요하지만, 제안 방법은 텍스트 프롬프트만으로 초기화를 수행할 수 있으며, 영상 중간에 사용자가 새로운 포인트를 추가하여 객체를 쉽게 지정할 수 있다.
- **파인튜닝의 중요성**: Ablation study를 통해 단순한 경량 SAM보다 수술 데이터로 파인튜닝된 모델의 성능이 비약적으로 향상됨을 확인하였다. 이는 ViT-H 기반의 거대 SAM 모델보다도 더 나은 결과를 보여준다.

### 한계 및 비판적 해석

- **특수 환경 취약성**: UCL dVRK 데이터셋의 저조도 환경이나 SAR-RARP 데이터셋의 심한 출혈, 급격한 도구 움직임, 카메라 포커스 변경 등이 발생하는 복잡한 장면에서는 성능이 저하되는 한계가 있다.
- **가정**: 포인트 트래커가 포인트의 위치를 정확히 유지한다는 가정하에 분할이 이루어지므로, 트래커의 실패가 곧 분할의 실패로 이어진다.

## 📌 TL;DR

본 논문은 **CoTracker(포인트 트래커)**와 **Fine-tuned MobileSAM(경량 분할 모델)**을 결합하여, 수술 도구 분할을 실시간으로 수행하는 프레임워크를 제안하였다. 특히 경량 SAM을 수술 도메인에 맞춰 포인트 프롬프트 기반으로 파인튜닝함으로써 효율성과 정확도를 동시에 잡았으며, RTX 4090 기준 **90 FPS**라는 매우 빠른 속도를 달성하였다. 이 연구는 향후 실시간 수술 가이드 시스템이나 수술 자동화 연구에서 강력한 VOS 베이스라인으로 활용될 가능성이 높다.
