# 2nd Place Solution for MOSE Track in CVPR 2024 PVUW workshop: Complex Video Object Segmentation

Zhensong Xu, Jiangtao Yao, Chengjing Wu, Ting Liu and Luoqi Liu (2024)

## 🧩 Problem to Solve

본 논문은 CVPR 2024의 PVUW(Pixel-level Video Understanding in the Wild) 워크숍 내 MOSE 트랙에서 제안된 2위 솔루션에 대해 다룬다. 해결하고자 하는 핵심 문제는 복잡한 환경에서의 반지도 비디오 객체 분할(Semi-supervised Video Object Segmentation, VOS)이다.

특히 MOSE 데이터셋은 기존의 DAVIS나 YouTubeVOS와 같은 데이터셋에 비해 다음과 같은 까다로운 특성을 가지고 있다.

- **작은 객체(Tiny objects)와 유사한 객체(Similar objects):** 모델이 객체를 혼동하거나 정밀하게 분할하는 데 어려움을 준다.
- **빠른 움직임(Fast movements):** 연속된 프레임 간의 추적 성능을 저하시킨다.
- **높은 소멸-재등장률(Disappearance-reappearance rates) 및 심한 폐색(Heavy occlusions):** 객체가 사라졌다가 다시 나타나거나 다른 객체에 의해 가려지는 빈도가 높아 추적의 연속성을 유지하기 어렵다.

따라서 본 연구의 목표는 이러한 복잡한 환경에서도 강건하게 작동하는 VOS 파이프라인을 구축하여 분할 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 강력한 베이스라인 모델인 Cutie를 바탕으로, 데이터 증강(Data Augmentation)과 추론 전략(Inference Strategy)을 최적화하여 모델의 시맨틱 표현력과 강건성을 향상시키는 것이다.

1. **시맨틱 표현력 강화를 위한 데이터 증강:** Mask2Former를 이용해 MOSE의 검증 및 테스트 세트에서 인스턴스 마스크를 추출하고, COCO 데이터셋의 특정 클래스를 활용하여 사전 학습(Pretraining) 데이터를 확장함으로써 작은 객체와 유사 객체에 대한 판별력을 높였다.
2. **모션 블러(Motion Blur) 도입:** 학습 과정에 무작위 커널 크기와 각도의 모션 블러를 추가하여, 실제 비디오에서 발생하는 움직임으로 인한 이미지 흐림 현상에 대한 강건성을 확보하였다.
3. **추론 단계의 최적화:** 테스트 시간 증강(Test Time Augmentation, TTA)과 메모리 전략(Memory Strategy)을 적용하여 최종 성능을 극대화하였다.

## 📎 Related Works

본 논문은 메모리 기반(Memory-based) VOS 접근 방식을 채택하고 있다. 메모리 기반 방식은 과거의 분할된 프레임을 메모리 뱅크에 저장하고, 새로운 쿼리 프레임이 들어오면 Cross-attention을 통해 메모리를 읽어오는 방식으로, 표류(Drifting) 및 폐색 문제에 강건하다.

- **STM (Space-Time Memory network):** 과거 프레임과 마스크를 메모리에 저장하고 픽셀 수준의 매칭을 수행한 초기 성공 사례이다.
- **STCN (Space-Time Correspondence Network):** STM을 발전시켜 마스크 없이 프레임의 키 특징(Key features)만을 인코딩하고 $L2$ 유사도를 사용하여 효율성과 효과를 높였다.
- **XMem:** 인간의 기억 모델(Atkinson–Shiffrin model)에서 영감을 얻어 Sensory, Working, Long-term memory라는 3단계 메모리 뱅크를 도입하여 긴 비디오에서도 우수한 성능을 보인다.
- **Cutie:** 객체 메모리(Object memory)와 객체 트랜스포머(Object transformer)를 통해 양방향 정보 상호작용을 수행하며, 폐색과 유사성이 심한 어려운 장면에서 매우 강건한 성능을 보인다.

본 연구에서는 이러한 최신 기법 중 가장 성능이 우수한 **Cutie**를 베이스라인으로 선정하여 확장하였다.

## 🛠️ Methodology

### 1. 베이스라인 아키텍처: Cutie

Cutie는 고해상도 픽셀 메모리 $F$와 고수준 객체 메모리 $S$를 동시에 유지한다.

- **픽셀 메모리 ($F$):** 메모리 프레임과 해당 분할 마스크로부터 인코딩된다.
- **객체 메모리 ($S$):** 메모리 프레임에서 객체 수준의 특징을 압축하여 저장한다.

작동 과정은 다음과 같다. 쿼리 프레임이 입력되면 먼저 픽셀 메모리에서 픽셀 리드아웃(Pixel readout) $R_0$를 추출한다. 이후 $R_0$는 객체 메모리 및 학습 가능한 객체 쿼리(Object queries) $X$와 함께 **Object Transformer** 블록을 통과하며 양방향으로 상호작용한다.

- **Bottom-up:** 픽셀 리드아웃 $\rightarrow$ 객체 쿼리 (Masked cross attention 사용)
- **Top-down:** 객체 쿼리 $\rightarrow$ 픽셀 리드아웃 (Cross attention 사용)

최종적으로 업데이트된 픽셀 리드아웃 $R_l$은 디코더에서 스킵 연결(Skip connections)을 통해 전달된 다중 스케일 특징과 결합되어 최종 마스크를 생성한다.

### 2. 데이터 증강 (Data Augmentation)

Cutie의 2단계 학습 파이프라인(사전 학습 $\rightarrow$ 메인 학습)에 다음 전략을 적용하였다.

- **인스턴스 분할 기반 사전 학습 데이터 생성:**
  - MOSE의 valid 및 test 세트의 이미지에 Mask2Former를 적용하여 인스턴스 마스크를 생성하였다. 이는 MOSE 데이터셋 특유의 객체 외형을 모델이 미리 학습하게 하여 작은 객체 분할 성능을 높인다.
  - COCO 데이터셋에서 MOSE에 자주 등장하는 클래스(사람, 동물, 차량)를 선택하여 독립적인 이진 마스크(Binary masks)로 변환해 추가하였다.
- **모션 블러 (Motion Blur):**
  - 학습 과정(사전 학습 및 메인 학습 모두)에서 무작위 커널 크기와 각도를 가진 모션 블러를 적용하여, 빠른 움직임으로 인해 발생하는 이미지 흐림에 대비하였다.

### 3. 추론 단계 최적화 (Inference Operations)

- **TTA (Test Time Augmentation):**
  - **Horizontal Flipping:** 수평 뒤집기를 적용하여 결과를 앙상블하였다.
  - **Multi-scale Enhancement:** 짧은 쪽 변의 해상도를 $600\text{p}$, $720\text{p}$, $800\text{p}$의 세 가지 설정으로 추론한 후 그 결과를 평균 내어 최종 마스크를 생성하였다.
- **메모리 전략 (Memory Strategy):**
  - 메모리 뱅크의 크기가 크고 메모리 간격이 짧을수록 성능이 향상됨을 확인하였다. 이에 따라 최대 메모리 프레임 수 $T_{\max}$를 18로 설정하고, 메모리 간격(Memory interval)을 1로 조정하였다.

## 📊 Results

### 실험 설정

- **사전 학습 데이터:** ECSSD, DUTS, FSS-1000, HRSOD, BIG 및 Mask2Former로 생성한 MOSE/COCO 데이터(각각 66,823개 및 89,490개 쌍)를 사용하였다.
- **메인 학습 데이터:** DAVIS-2017, YouTubeVOS-2019, BURST, OVIS, MOSE를 혼합하여 사용하였다.
- **학습 하이퍼파라미터:** AdamW 옵티마이저, 학습률 $0.0001$, 배치 사이즈 16, 가중치 감쇠(Weight decay) $0.001$을 사용하였다.

### 정량적 결과

MOSE 챌린지 리더보드 결과, 제안 방법은 $\text{aJ} = 0.8007$, $\text{aF} = 0.8683$, $\text{aJ\&F} = 0.8345$를 기록하며 **전체 2위**를 차지하였다.

### 절제 연구 (Ablation Study)

각 구성 요소의 기여도는 다음과 같다 (J&F 지표 기준).

| 방법 | J&F | 향상분 |
| :--- | :---: | :---: |
| Baseline (Cutie) | 0.7857 | - |
| Baseline + DA (Data Augmentation) | 0.8043 | $+0.0186$ |
| Baseline + DA + TTA + MS | $\mathbf{0.8345}$ | $+0.0488$ |

분석 결과, 인스턴스 분할 및 모션 블러를 통한 데이터 증강이 성능을 향상시켰으며, 특히 TTA와 메모리 전략(MS)이 가장 큰 성능 향상을 가져왔음을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 모델 아키텍처의 근본적인 변경보다는 **데이터의 질적 향상과 추론 전략의 세밀한 튜닝**이 실제 챌린지 환경에서 얼마나 중요한지를 보여준다.

- **강점:** Mask2Former를 이용해 타겟 데이터셋(MOSE)의 특성을 반영한 사전 학습 데이터를 생성한 점이 매우 영리한 접근이다. 이는 도메인 간의 간극(Domain gap)을 줄여 작은 객체에 대한 시맨틱 표현력을 직접적으로 높이는 효과를 주었다.
- **한계 및 논의:**
  - TTA와 다중 스케일 추론은 성능을 크게 높이지만, 추론 시간을 증가시켜 실시간 적용에는 제약이 있을 수 있다.
  - 메모리 프레임 수 $T_{\max}$를 늘리는 것이 성능 향상을 가져왔으나, 이는 메모리 사용량 증가와 연산 비용 상승을 초래한다.
  - 본 논문은 특정 챌린지의 순위를 높이기 위한 '솔루션' 성격이 강하므로, 일반화된 새로운 이론적 기여보다는 실무적인 최적화 기법의 조합에 집중되어 있다.

## 📌 TL;DR

본 논문은 복잡한 비디오 객체 분할(VOS) 작업인 MOSE 트랙에서 2위를 차지한 솔루션을 제안한다. 강력한 베이스라인인 **Cutie**에 **Mask2Former 기반의 시맨틱 사전 학습 데이터 확장**, **모션 블러 증강**, 그리고 **TTA 및 메모리 뱅크 최적화**를 결합하여 작은 객체, 유사 객체, 빠른 움직임이라는 난제를 해결하였다. 이 연구는 데이터셋 특성에 맞춘 정교한 데이터 증강과 추론 전략이 VOS 성능 향상에 결정적인 역할을 함을 시사한다.
