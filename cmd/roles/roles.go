// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package roles

import (
	"context"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	rocksmqimpl "github.com/milvus-io/milvus/internal/mq/mqimpl/rocksmq/server"

	"go.uber.org/zap"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/milvus-io/milvus/cmd/components"
	"github.com/milvus-io/milvus/internal/indexnode"
	"github.com/milvus-io/milvus/internal/log"
	"github.com/milvus-io/milvus/internal/metrics"
	"github.com/milvus-io/milvus/internal/util/etcd"
	"github.com/milvus-io/milvus/internal/util/healthz"
	"github.com/milvus-io/milvus/internal/util/metricsinfo"
	"github.com/milvus-io/milvus/internal/util/paramtable"
	"github.com/milvus-io/milvus/internal/util/trace"
	"github.com/milvus-io/milvus/internal/util/typeutil"
)

var Params paramtable.ComponentParam

// all milvus related metrics is in a separate registry
var Registry *prometheus.Registry

func init() {
	Registry = prometheus.NewRegistry()
	Registry.MustRegister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))
	Registry.MustRegister(prometheus.NewGoCollector())
}

func stopRocksmq() {
	rocksmqimpl.CloseRocksMQ()
}

// MilvusRoles decides which components are brought up with Milvus.
type MilvusRoles struct {
	HasMultipleRoles bool
	EnableRootCoord  bool `env:"ENABLE_ROOT_COORD"`
	EnableProxy      bool `env:"ENABLE_PROXY"`
	EnableQueryCoord bool `env:"ENABLE_QUERY_COORD"`
	EnableQueryNode  bool `env:"ENABLE_QUERY_NODE"`
	EnableDataCoord  bool `env:"ENABLE_DATA_COORD"`
	EnableDataNode   bool `env:"ENABLE_DATA_NODE"`
	EnableIndexCoord bool `env:"ENABLE_INDEX_COORD"`
	EnableIndexNode  bool `env:"ENABLE_INDEX_NODE"`
}

// EnvValue not used now.
func (mr *MilvusRoles) EnvValue(env string) bool {
	env = strings.ToLower(env)
	env = strings.Trim(env, " ")
	return env == "1" || env == "true"
}

func (mr *MilvusRoles) printLDPreLoad() {
	const LDPreLoad = "LD_PRELOAD"
	val, ok := os.LookupEnv(LDPreLoad)
	if ok {
		log.Info("Enable Jemalloc", zap.String("Jemalloc Path", val))
	}
}

// func (mr *MilvusRoles) runIndexCoord(ctx context.Context, localMsg bool) *components.IndexCoord {
// var is *components.IndexCoord
// var wg sync.WaitGroup

// wg.Add(1)
// go func() {
// 	indexcoord.Params.InitOnce()
// 	if localMsg {
// 		indexcoord.Params.SetLogConfig(typeutil.StandaloneRole)
// 	} else {
// 		indexcoord.Params.SetLogConfig(typeutil.IndexCoordRole)
// 	}

// 	factory := dependency.NewFactory(localMsg)

// 	var err error
// 	is, err = components.NewIndexCoord(ctx, factory)
// 	if err != nil {
// 		panic(err)
// 	}
// 	if !mr.HasMultipleRoles {
// 		http.Handle(healthz.HealthzRouterPath, &componentsHealthzHandler{component: is})
// 	}
// 	wg.Done()
// 	_ = is.Run()
// }()
// wg.Wait()

// metrics.RegisterIndexCoord(Registry)
// return is
// 	return nil
// }

func (mr *MilvusRoles) runIndexNode(ctx context.Context, localMsg bool, alias string) *components.IndexNode {
	var in *components.IndexNode
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		indexnode.Params.IndexNodeCfg.InitAlias(alias)
		indexnode.Params.InitOnce()
		if localMsg {
			indexnode.Params.SetLogConfig(typeutil.StandaloneRole)
		} else {
			indexnode.Params.SetLogConfig(typeutil.IndexNodeRole)
		}

		// factory := dependency.NewFactory(localMsg)

		var err error
		in, err = components.NewIndexNode(ctx)
		if err != nil {
			panic(err)
		}
		if !mr.HasMultipleRoles {
			http.Handle(healthz.HealthzRouterPath, &componentsHealthzHandler{component: in})
		}
		wg.Done()
		_ = in.Run()
	}()
	wg.Wait()

	metrics.RegisterIndexNode(Registry)
	return in
}

// Run Milvus components.
func (mr *MilvusRoles) Run(local bool, alias string) {
	log.Info("starting running Milvus components")
	ctx, cancel := context.WithCancel(context.Background())
	mr.printLDPreLoad()

	// only standalone enable localMsg
	if local {
		if err := os.Setenv(metricsinfo.DeployModeEnvKey, metricsinfo.StandaloneDeployMode); err != nil {
			log.Error("Failed to set deploy mode: ", zap.Error(err))
		}
		Params.Init()

		if Params.EtcdCfg.UseEmbedEtcd {
			// Start etcd server.
			etcd.InitEtcdServer(&Params.EtcdCfg)
			defer etcd.StopEtcdServer()
		}
	} else {
		if err := os.Setenv(metricsinfo.DeployModeEnvKey, metricsinfo.ClusterDeployMode); err != nil {
			log.Error("Failed to set deploy mode: ", zap.Error(err))
		}
	}

	if os.Getenv(metricsinfo.DeployModeEnvKey) == metricsinfo.StandaloneDeployMode {
		closer := trace.InitTracing("standalone")
		if closer != nil {
			defer closer.Close()
		}
	}

	// var is *components.IndexCoord
	// if mr.EnableIndexCoord {
	// 	is = mr.runIndexCoord(ctx, local)
	// 	if is != nil {
	// 		defer is.Stop()
	// 	}
	// }

	var in *components.IndexNode
	if mr.EnableIndexNode {
		in = mr.runIndexNode(ctx, local, alias)
		if in != nil {
			defer in.Stop()
		}
	}

	metrics.ServeHTTP(Registry)
	sc := make(chan os.Signal, 1)
	signal.Notify(sc,
		syscall.SIGHUP,
		syscall.SIGINT,
		syscall.SIGTERM,
		syscall.SIGQUIT)
	sig := <-sc
	log.Error("Get signal to exit\n", zap.String("signal", sig.String()))

	// some deferred Stop has race with context cancel
	cancel()
}
