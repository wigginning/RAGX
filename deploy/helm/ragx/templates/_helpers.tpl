{{/* Common helpers for the RAGX chart. */}}

{{- define "ragx.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ragx.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "ragx.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels attached to every object.
*/}}
{{- define "ragx.labels" -}}
helm.sh/chart: {{ include "ragx.chart" . }}
{{ include "ragx.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: ragx
{{- end -}}

{{/*
Selector labels — must be stable (no version) so rolling updates work.
*/}}
{{- define "ragx.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ragx.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "ragx.apiLabels" -}}
{{ include "ragx.labels" . }}
app.kubernetes.io/component: api
{{- end -}}

{{- define "ragx.workerLabels" -}}
{{ include "ragx.labels" . }}
app.kubernetes.io/component: worker
{{- end -}}

{{- define "ragx.mockLlmLabels" -}}
{{ include "ragx.labels" . }}
app.kubernetes.io/component: mock-llm
{{- end -}}

{{- define "ragx.prometheusLabels" -}}
{{ include "ragx.labels" . }}
app.kubernetes.io/component: prometheus
{{- end -}}

{{- define "ragx.grafanaLabels" -}}
{{ include "ragx.labels" . }}
app.kubernetes.io/component: grafana
{{- end -}}

{{- define "ragx.secretName" -}}
{{ include "ragx.fullname" . }}-secrets
{{- end -}}

{{- define "ragx.apiEnvConfigMap" -}}
{{ include "ragx.fullname" . }}-api-env
{{- end -}}

{{- define "ragx.prometheusConfigMap" -}}
{{ include "ragx.fullname" . }}-prometheus
{{- end -}}

{{- define "ragx.grafanaProvisioningConfigMap" -}}
{{ include "ragx.fullname" . }}-grafana-provisioning
{{- end -}}

{{/*
middlewareEnv renders the RAGX_* connection env vars for the API/worker pods.
It is data-dependent: each block is only emitted when the matching component
is enabled, so the same snippet works for both lite (no middleware) and full.
Passwords are sourced from the chart-managed (or external) Secret and surfaced
as $(_RAGX_*_PASSWORD) variable references so the URL envs stay templated.
*/}}
{{- define "ragx.middlewareEnv" -}}
{{- $full := include "ragx.fullname" . -}}
{{- $secret := include "ragx.secretName" . -}}
{{- if .Values.elasticsearch.enabled }}
        - name: RAGX_PLUGINS.VECTOR_STORE.ES.HOSTS
          value: "http://{{ $full }}-elasticsearch:{{ .Values.elasticsearch.port }}"
        - name: RAGX_PLUGINS.VECTOR_STORE.ES.INDEX
          value: "ragx_chunks"
{{- end }}
{{- if .Values.neo4j.enabled }}
        - name: _RAGX_NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: {{ $secret }}
              key: neo4j-password
        - name: RAGX_PLUGINS.GRAPH_STORE.NEO4J.URI
          value: "bolt://{{ $full }}-neo4j:{{ .Values.neo4j.ports.bolt }}"
        - name: RAGX_PLUGINS.GRAPH_STORE.NEO4J.USER
          value: "neo4j"
        - name: RAGX_PLUGINS.GRAPH_STORE.NEO4J.DATABASE
          value: "neo4j"
        - name: RAGX_PLUGINS.GRAPH_STORE.NEO4J.PASSWORD
          value: "$(_RAGX_NEO4J_PASSWORD)"
{{- end }}
{{- if .Values.minio.enabled }}
        - name: _RAGX_MINIO_ACCESS
          valueFrom:
            secretKeyRef:
              name: {{ $secret }}
              key: minio-access-key
        - name: _RAGX_MINIO_SECRET
          valueFrom:
            secretKeyRef:
              name: {{ $secret }}
              key: minio-secret-key
        - name: RAGX_PLUGINS.OBJECT_STORE.MINIO.ENDPOINT
          value: "{{ $full }}-minio:{{ .Values.minio.port }}"
        - name: RAGX_PLUGINS.OBJECT_STORE.MINIO.ACCESS_KEY
          value: "$(_RAGX_MINIO_ACCESS)"
        - name: RAGX_PLUGINS.OBJECT_STORE.MINIO.SECRET_KEY
          value: "$(_RAGX_MINIO_SECRET)"
        - name: RAGX_PLUGINS.OBJECT_STORE.MINIO.BUCKET
          value: {{ .Values.minio.bucket | quote }}
{{- end }}
{{- if .Values.redis.enabled }}
        - name: RAGX_PLUGINS.CACHE.REDIS.URL
          value: "redis://{{ $full }}-redis:{{ .Values.redis.port }}/0"
        - name: RAGX_QUEUE.DRIVER
          value: "redis"
        - name: RAGX_QUEUE.REDIS_URL
          value: "redis://{{ $full }}-redis:{{ .Values.redis.port }}/0"
{{- end }}
{{- if .Values.postgresql.enabled }}
        - name: _RAGX_PG_PASSWORD
          valueFrom:
            secretKeyRef:
              name: {{ $secret }}
              key: postgres-password
        - name: RAGX_METADATA_DB.URL
          value: "postgresql://{{ .Values.postgresql.user }}:$(_RAGX_PG_PASSWORD)@{{ $full }}-postgresql:{{ .Values.postgresql.port }}/{{ .Values.postgresql.database }}"
{{- end }}
{{- if .Values.mockLlm.enabled }}
        - name: RAGX_LLM.PROVIDERS.OPENAI_COMPAT.BASE_URL
          value: "http://{{ $full }}-mock-llm:{{ .Values.mockLlm.port }}/v1"
{{- end }}
{{- end -}}
