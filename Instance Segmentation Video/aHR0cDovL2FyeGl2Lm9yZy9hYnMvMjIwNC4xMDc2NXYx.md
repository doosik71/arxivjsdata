# Tag-Based Attention Guided Bottom-Up Approach for Video Instance Segmentation

Jyoti Kini and Mubarak Shah (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **Video Instance Segmentation (VIS)**이다. VIS는 비디오 시퀀스 전체에서 각 객체 인스턴스를 픽셀 단위로 분할(Segmentation)함과 동시에, 서로 다른 프레임 간에 동일한 객체를 식별하여 추적(Tracking)하는 복합적인 작업이다.

기존의 VIS 접근 방식은 주로 **Top-down 방식**을 채택하고 있다. 이는 객체 탐지(Detection), 분할(Segmentation), 그리고 추적(Tracking)을 각각 별도의 네트워크나 단계로 처리하는 다단계 구조이다. 그러나 이러한 방식은 다음과 같은 한계점을 가진다.
1. **훈련의 복잡성**: 서로 분리된 모듈들을 연결하여 학습시켜야 하므로 최적의 솔루션을 찾기 어렵고 학습 과정이 번거롭다.
2. **탐지기 의존성**: Mask R-CNN과 같은 지역 제안(Region Proposal) 기반 방식은 탐지기가 실패할 경우 전체 시스템의 성능이 급격히 저하된다. 특히 배경이 복잡하거나 폐색(Occlusion)이 심한 환경에서 취약하다.
3. **중복성 및 시간 정보 부족**: 많은 수의 중복된 지역 제안이 생성되어 프레임 간 연관성 계산이 복잡해지며, 프레임 단위로 처리하는 특성상 비디오의 전역적인 시간적 맥락(Temporal Context)을 충분히 활용하지 못한다.

따라서 본 연구의 목표는 지역 제안 과정 없이 픽셀 수준에서 인스턴스를 직접 구분하는 **End-to-end 학습 가능한 Bottom-up 방식**의 VIS 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 VIS 작업을 **태그 할당 문제(Tag Assignment Problem)**로 정의하는 것이다. 즉, 각 픽셀에 0과 1 사이의 임의의 태그 값을 할당하고, 동일한 인스턴스에 속한 픽셀들은 유사한 태그 값을, 서로 다른 인스턴스에 속한 픽셀들은 확연히 다른 태그 값을 갖도록 학습시키는 것이다.

주요 기여 사항은 다음과 같다.
- **Bottom-up 접근 방식**: Region Proposal 없이 픽셀 임베딩만으로 VIS를 해결하여 훈련 복잡도를 낮추고 속도를 향상시켰다.
- **3D Volume 처리**: 비디오를 프레임 단위가 아닌 단일 3D 볼륨으로 처리하여 시간적 정보를 직접적으로 통합하였다.
- **Spatio-Temporal Tagging Loss**: 인스턴스 간의 분리성과 동일 인스턴스의 일관성을 보장하기 위한 새로운 손실 함수를 제안하였다.
- **Tag-based Attention Module**: 생성된 태그를 기반으로 인스턴스 태그를 정교화하고, 비디오 전체에서 인스턴스 마스크의 전파(Propagation)를 학습하는 모듈을 도입하였다.
- **VSS 보조 작업**: Video Semantic Segmentation (VSS)을 보조 작업으로 수행하여, 여기서 발생하는 그래디언트 전파가 주 작업인 VIS의 성능을 향상시키도록 설계하였다.

## 📎 Related Works

### Video Semantic Segmentation (VSS)
VSS는 비디오 내 모든 픽셀에 클래스 레이블을 부여하는 작업이다. 동일 클래스의 모든 객체는 같은 레이블을 갖지만, 개별 인스턴스를 구분하거나 추적하는 기능은 없다.

### Video Object Segmentation (VOS)
VOS는 첫 프레임에서 주어진 정답 마스크를 바탕으로 특정 객체 인스턴스를 비디오 전체에서 추적하며 분할하는 작업이다. 클래스 구분보다는 특정 객체의 마스크 전파에 집중한다.

### Video Instance Segmentation (VIS)
VIS는 VSS와 VOS의 특성을 모두 포함한다. 기존의 MaskTrack R-CNN, MaskProp 등은 Top-down 방식을 사용하여 높은 성능을 냈으나, 앞서 언급한 복잡성과 탐지기 의존성 문제가 있다. 최근 STEm-Seg와 같은 Bottom-up 방식이 등장했으나, 임베딩 공간에서 인스턴스 클러스터 간의 분리도가 낮아 성능이 제한적이었다. 본 논문은 이를 **태그 기반 어텐션**과 **시공간 태깅 손실**을 통해 해결하고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조
본 모델은 입력 비디오 클립(32프레임)을 처리하여 인스턴스 태그 $p_t$와 시맨틱 세그멘테이션 마스크 $O_{t,c}$를 동시에 생성한다. 전체 파이프라인은 다음과 같다.
1. **Encoder**: $\text{ResNet(2+1)D}$를 사용하여 비디오 클립의 특징을 추출한다.
2. **Spatio-Temporal Attention**: 추출된 특징에 시공간적 어텐션을 적용하여 장기적 시간 의존성과 공간적 맥락을 강화한 임베딩 $u_t$를 생성한다.
3. **Tag Generator**: $u_t$를 통해 각 프레임의 픽셀별 인스턴스 태그 $p_t$를 생성한다.
4. **Tag-based Attention**: 생성된 태그 $p_t$를 입력받아 태그 간의 연관성을 학습하고, 정제된 임베딩 $w_t$를 출력한다.
5. **Self-Attention**: 입력 특징 $q_t$로부터 비지역적(non-local) 맥락을 캡처한 특징 $v_t$를 생성한다.
6. **Decoder**: $v_t$와 $w_t$를 결합(Concatenate)하여 최종 시맨틱 세그멘테이션 마스크 $O_{t,c}$를 출력한다.

### 주요 모듈 설명
- **Spatio-Temporal Attention**: 쿼리(Q)는 모든 프레임의 특징 $f_{t=1 \dots 32}$로, 키(K)와 밸류(V)는 첫 번째 프레임의 특징 $f_{t=1}$로 설정하여 프레임 간의 맥락을 통합한다.
- **Tag-based Attention**: 예측된 태그들의 전역적 맥락을 활용하여 각 픽셀의 태그 값을 보정함으로써 인스턴스 간의 분리도를 높인다.
- **Self-Attention**: VSS 보조 작업을 위해 프레임 내의 풍부한 특징 표현을 제공하며, 이는 결과적으로 VIS 성능 향상으로 이어진다.

### 훈련 목표 및 손실 함수
본 모델은 시맨틱 세그멘테이션을 위한 $\mathcal{L}_{\text{crossentropy}}$와 인스턴스 구분을 위한 $\mathcal{L}_{\text{tag}}$를 동시에 최적화한다.

#### Spatio-Temporal Tagging Loss ($\mathcal{L}_{\text{tag}}$)
인스턴스 $n$의 $m$번째 픽셀의 예측 태그 값을 $h'_{nm}$, 해당 인스턴스의 평균 태그 값을 $h_n = \frac{1}{M} \sum_{m=1}^{M} h'_{nm}$이라고 할 때, 네 가지 손실의 합으로 정의된다.

1. **Spatial-intra-instance loss ($\mathcal{L}_{\text{spectra}}$)**: 동일 인스턴스 내의 픽셀들이 평균 태그 값에 가까워지도록 유도한다.
   $$\mathcal{L}_{\text{spectra}} = \frac{1}{N} \sum_{n=1}^{N} \sum_{m=1}^{M'} (h_n - h'_{nm})^2$$
2. **Spatial-inter-instance loss ($\mathcal{L}_{\text{specter}}$)**: 서로 다른 인스턴스 간의 태그 값 차이가 최소 마진 $G$ 이상이 되도록 밀어낸다.
   $$\mathcal{L}_{\text{specter}} = \sum_{n=1}^{N-1} \sum_{n'=n+1}^{N} \max(0, G - \|h_n - h_{n'}\|)$$
3. **Temporal-instance-grouping loss ($\mathcal{L}_{\text{tempra}}$)**: 동일 인스턴스가 시간축($T$)을 따라 일관된 태그 값을 갖도록 강제한다.
   $$\mathcal{L}_{\text{tempra}} = \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{T} (h_n - h_{nt})^2$$
4. **Temporal-instance-separation loss ($\mathcal{L}_{\text{temper}}$)**: 시간축 상에서 서로 다른 인스턴스들이 충분히 분리된 태그 값을 갖도록 한다.
   $$\mathcal{L}_{\text{temper}} = \sum_{n=1}^{N-1} \sum_{n'=n+1}^{N'} \sum_{t=1}^{T} \max(0, G - \|h_{nt} - h_{n't}\|)$$

최종 손실 함수는 다음과 같다.
$$\mathcal{L}_{\text{overall}} = (\mathcal{L}_{\text{spectra}} + \mathcal{L}_{\text{specter}} + \mathcal{L}_{\text{tempra}} + \mathcal{L}_{\text{temper}}) + \mathcal{L}_{\text{crossentropy}}$$

## 📊 Results

### 실험 설정
- **데이터셋**: YouTube-VIS 및 DAVIS'19 Unsupervised 데이터셋을 사용하였다.
- **학습 세부사항**: $\text{ResNet(2+1)D}$ 백본은 Kinetics-400으로 사전 학습되었으며, Adam Optimizer(LR=0.0001)를 사용하여 100 에포크 동안 학습하였다.
- **추론**: 32프레임 단위의 클립으로 처리하며, 누락된 중간 프레임들은 마스크 보간법(Interpolation)을 통해 채운다.

### 정량적 결과
- **YouTube-VIS**: 제안 방법은 mAP 57.9, AP@50 74.8 등의 성능을 기록하며 MaskProp나 Propose-Reduce와 같은 최신 Top-down 방식들과 경쟁 가능한 수준의 성능을 보였다.
- **DAVIS'19**: J&F Mean 지표에서 76.4를 기록하며 기존의 STEm-Seg(67.8)나 UnOVOST(68.4)보다 월등한 성능을 보였다. 특히 추론 속도(FPS) 면에서 효율적인 트레이드-오프를 달성하였다.

### 절제 연구 (Ablation Study)
- **손실 함수의 영향**: $\mathcal{L}_{\text{spectra}}$가 없으면 인스턴스가 작은 조각으로 분리되고, $\mathcal{L}_{\text{specter}}$가 없으면 서로 다른 인스턴스의 태그가 겹치는 현상이 발생하였다. 시간 관련 손실($\mathcal{L}_{\text{tempra}}, \mathcal{L}_{\text{temper}}$)은 프레임 간 태그 일관성에 결정적인 역할을 하였다.
- **어텐션 모듈의 영향**: Spatio-Temporal Attention은 장기 의존성 캡처에, Tag-based Attention은 인스턴스 간 분리도 및 전파 일관성 향상에 기여하였다.
- **보조 작업의 영향**: VSS 브랜치와 관련 커넥터를 제거했을 때 VIS 성능이 크게 하락하였다. 이는 보조 작업으로부터 오는 그래디언트 전파가 VIS의 특징 학습을 돕는다는 것을 입증한다.

## 🧠 Insights & Discussion

본 논문은 기존 VIS 연구들이 집착하던 "탐지 후 추적"이라는 Top-down 패러다임에서 벗어나, **"태그 할당"이라는 Bottom-up 방식**으로도 충분히 경쟁력 있는 성능을 낼 수 있음을 보여주었다.

**강점:**
- **단순성 및 효율성**: 복잡한 다단계 파이프라인과 외부 네트워크 의존성을 제거하여 시스템을 단순화하였고, 이는 곧 빠른 추론 속도로 이어졌다.
- **시공간적 통합**: 비디오를 3D 볼륨으로 처리함으로써 프레임 간의 시간적 맥락을 직접적으로 학습에 활용하였다.
- **상호 보완적 설계**: 태그 기반의 손실 함수와 어텐션 모듈, 그리고 VSS라는 보조 작업이 서로 유기적으로 연결되어 픽셀 수준의 임베딩 품질을 높였다.

**한계 및 논의사항:**
- **VSS 성능의 한계**: 저자들은 VSS 결과가 Reasonable 하지만, VIS에 비해 정교하지 않다고 언급하였다. 이는 VSS를 위한 전용 손실 함수가 부족했기 때문이며, 향후 VSS 전용 손실을 추가한다면 두 작업 모두 성능이 향상될 가능성이 크다.
- **보간법 의존성**: 추론 시 32프레임 단위로 처리하고 나머지 프레임을 보간법으로 채우는 방식은 매우 빠른 움직임이 있는 영상에서 정밀도가 떨어질 수 있다.

## 📌 TL;DR

본 논문은 Region Proposal 없이 픽셀에 고유한 **태그(Tag)**를 부여하여 객체를 구분하고 추적하는 **Bottom-up 방식의 Video Instance Segmentation** 프레임워크를 제안한다. $\text{ResNet(2+1)D}$ 기반의 시공간 특징 추출과 정교한 **Spatio-Temporal Tagging Loss**, 그리고 **Tag-based Attention**을 통해 인스턴스 분리도와 시간적 일관성을 확보하였다. 특히 VSS(비디오 시맨틱 세그멘테이션)를 보조 작업으로 활용하여 성능을 극대화하였으며, 기존 Top-down 방식보다 훨씬 빠른 속도로 경쟁력 있는 정확도를 달성하였다. 이 연구는 복잡한 탐지-추적 파이프라인 없이도 픽셀 임베딩 최적화만으로 VIS 문제를 해결할 수 있는 새로운 방향성을 제시한다.