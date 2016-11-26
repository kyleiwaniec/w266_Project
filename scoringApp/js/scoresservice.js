iq4scoring.service('ScoresService', ScoresService);
ScoresService.$inject = ['$http','$stateParams'];
function ScoresService($http, $stateParams){
	var that = this;

	var topics = []
	that.it = makeIterator(topics);


    that.getTopics = function(){
		$http.get('/api/topics')
			.then(function(d){
				topics = d.data;
				that.it = makeIterator(topics);
				that.getTopic(that.it.next().value)
		})
	}

	that.getTopic = function(topic_id){
		$http.get('/api/topic/'+topic_id)
			.then(function(d){
				that.topic = d.data
				console.log('getTopic',topic_id)
				that.getComments(topic_id)
		})
	}

	that.getComments = function(topic_id){
		$http.get('/api/comments/'+topic_id)
			.then(function(d){
				that.comments = d.data
				console.log('getComments',topic_id)
		})
	}







	function makeIterator(array){
	    var nextIndex = -1;
	    return {
	       next: function(){
	           return nextIndex < array.length ?
	               {value: array[++nextIndex], done: false} :
	               {done: true};
	       },
	       prev: function(){
	           return nextIndex >= 0 ?
	               {value: array[--nextIndex], done: false} :
	               {done: true};
	       }
	    }
	}





}
