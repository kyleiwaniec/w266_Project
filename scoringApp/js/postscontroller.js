iq4scoring.controller('postsController', postsController);
postsController.$inject = ['$scope','$state', '$stateParams','$http','$window','ScoresService'];
function postsController($scope, $state, $stateParams,$http,$window,ScoresService) {

	
	$scope.ss = ScoresService;	

	ScoresService.getTopics()



	$scope.getNext = function(){
		ScoresService.getTopic(ScoresService.it.next().value)
	}
	$scope.getPrev = function(){
		ScoresService.getTopic(ScoresService.it.prev().value)
	}

	$scope.postScore = function(comment){
		comment.submitted = true
		console.log("score",comment.score)
	}


}