'use strict';

iq4scoring.config([
    '$locationProvider',
    '$stateProvider',
    '$urlRouterProvider',
    '$sceProvider',

    function($locationProvider, $stateProvider, $urlRouterProvider, $sceProvider) {

      
      $locationProvider.html5Mode(true);
      $urlRouterProvider.when('/', '/dashboard');
      $urlRouterProvider.otherwise("/dashboard");


      $stateProvider
        .state('dashboard', {
          url:'/dashboard',
          views: {
            'content': {
              templateUrl:  DOMAIN+'/views/posts.html',
              controller: "postsController"
            }
          }
        })
        .state('criteria', {
          url:'/criteria',
          views: {
            'content': {
              templateUrl: DOMAIN+'/views/criteria.html',
              controller: "criteriaController"
            }
          }
        })
        
    }
  ])
  .run(['$rootScope', '$location', '$state', '$stateParams',
    function($rootScope, $location, $state, $stateParams) {
      

      $rootScope.$state = $state;
      $rootScope.$on('$stateChangeStart', function() {
        //_log("$state.current start", $state.current)
        //_log("$stateParams start", $stateParams)
      })

      $rootScope.previousState;
      $rootScope.currentState;
      $rootScope.$on('$stateChangeSuccess', function(ev, to, toParams, from, fromParams) {
        $rootScope.previousState = from.name;
        $rootScope.currentState = to.name;
        // _log('Previous state:'+$rootScope.previousState)
        //_log('Current state:'+$rootScope.currentState)
      });

      // $rootScope.$on('$routeChangeSuccess', function(newRoute, oldRoute) {
      //     $location.hash($routeParams.scrollTo);
      //     $anchorScroll();
      //   });

      $rootScope.$on('$stateChangeError', function(event, toState, toParams, fromState, fromParams, error) {
        // _log("$state.current err", event, toState, toParams, fromState, fromParams, error)
        // _log("$state.current err", $state.current)
        // _log("$stateParams err", $stateParams)
      })


    }
  ]);
